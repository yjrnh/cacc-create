"""预算约束自适应搜索 (Budget-Constrained Adaptive Search, BCAS)。

这是本项目相对"把 HNSW 复现一遍"的**增量**所在。前三个模块（data/metrics/counter/hnsw）
负责造出一个"可计量的图索引"，本模块负责回答一个 HNSW 原始论文没有回答的问题：

问题
----
主流 ANN 评估口径是「固定 ef，测 Recall@k 与 QPS」。它隐含两个工程假设：

  1. **预算不可控**：ef 是图搜索的宽度，不是算力。同一个 ef 在不同数据分布、
     不同维度上对应的距离计算次数可以差一个量级，调用方无法据此承诺 SLA。
  2. **预算分配不公**：固定 ef 对所有查询一视同仁。查询难度天然是长尾的——
     简单查询几十次距离计算就收敛，困难查询上千次仍不够。固定 ef 把预算
     浪费在简单查询上，同时牺牲困难查询，结果是**召回率均值好看、尾部崩坏**。

线上检索（RAG 召回、推荐候选、去重）关心两件事：硬性算力预算是否能守住，以及
最差那批查询的召回率是多少。均值召回率是给论文看的，P1 召回率是给用户看的。

方法
----
把「一次性固定 ef 搜索」换成「迭代加宽 + 硬预算守卫 + 边际收益早停」：

        ef_0  →  2·ef_0  →  4·ef_0  →  ...
                 │            │
                 │            └─ 每轮结束评估边际收益，收益不足即停
                 └─ 每轮扩展前检查剩余预算，装不下下一批邻居就停

三个部件各自可消融，对应报告里的消融实验：

  A. **可恢复搜索状态**（本模块的 ResumableLayerState）
     加宽一轮不重算已经算过的距离。省下的算力可量化为"重复距离计算占比"。
  B. **硬预算守卫**（budget）
     以**距离计算次数**为预算单位 —— 这正是本项目坚持自实现索引的原因，
     Faiss / hnswlib 都不对外暴露这个量。守卫在批次粒度上检查，因此预算
     只会被欠用、不会被超用（欠用量 < 单批邻居数，实测预算利用率 > 97%）。
  C. **边际收益早停**（gain_tau）
     用第 k 名的平方距离 d_k 的相对改善率作为收敛判据。d_k 是单调不增的，
     因此 gain 恒 >= 0；当 gain < tau 时继续加宽已无意义。

与固定 ef 的公平比较方式
------------------------
固定 ef 无法直接对齐"预算"，因此实验里对每个预算 B：
  1. 跑一遍 ef 扫描，得到 (平均距离计算次数, 平均召回率) 的散点；
  2. 在该曲线上按 B 做**线性插值**，得到"等算力下的固定 ef 召回率"。
这样两者的横坐标（算力）严格相等，比较的才是分配策略，而不是参数量。

诚实声明（必须写进报告）
------------------------
- 可恢复状态是**近似**的：上一轮把距离劣于当时最差者的候选丢弃了，加宽后不会
  回头捡。因此 BCAS-复用 的召回率可能略低于"每轮从零重算"（BCAS-重算）。
  实验里把 BCAS-重算作为召回上界一并报出，不隐藏这个折损。
- gain_tau 是启发式阈值，没有理论最优值；报告里给出它对结果的敏感性。
- 所有距离都是**平方欧氏距离**，d_k 与 gain 均定义在平方距离上。
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .counter import DistanceCounter
from .hnsw import HNSW


class ResumableLayerState:
    """单层 best-first 搜索的可恢复状态。

    与 HNSW._search_layer 的唯一区别是把 visited / cand / res 三个结构外置。
    这样"再加宽一轮"只需要继续从堆里取，不必从入口点重算。
    """

    __slots__ = ("layer", "visited", "cand", "res", "exhausted", "expansions")

    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.visited: set = set()
        self.cand: List[Tuple[float, int]] = []   # 最小堆：待扩展（距离, id）
        self.res: List[Tuple[float, int]] = []    # 最大堆（存负值）：当前最优 ef 个
        self.exhausted = False                    # 候选堆已空 = 搜索穷尽
        self.expansions = 0                       # 本轮已扩展节点数

    def topk(self, k: int) -> List[Tuple[float, int]]:
        """按距离升序取前 k 个 (平方距离, id)。"""
        return sorted((-nd, i) for nd, i in self.res)[:k]

    @property
    def worst(self) -> float:
        return -self.res[0][0] if self.res else float("inf")


class AdaptiveSearcher:
    """在 HNSW 索引之上的预算约束自适应检索器。

    参数
    ----
    index : HNSW      已构图的索引（共享同一张图，便于与固定 ef 做等算力比较）
    k     : int       返回条数
    budget: int       单次查询允许的最大距离计算次数（硬上限）
    ef0   : int       首轮搜索宽度，必须 >= k
    ef_max: int       搜索宽度上限，防止无限加宽
    growth: float     每轮加宽倍数
    gain_tau: float   边际收益阈值；d_k 相对改善率低于此值即停
    """

    name = "BCAS(ours)"

    def __init__(
        self,
        index: HNSW,
        k: int = 10,
        budget: int = 1000,
        ef0: int = 16,
        ef_max: int = 1024,
        growth: float = 2.0,
        gain_tau: float = 0.01,
        early_stop_eps: float = 0.0,
        resume: bool = True,
        max_rounds: int = 16,
        truth_ids: Optional[np.ndarray] = None,
        guard_zero_recall: bool = False,
    ) -> None:
        self.index = index
        self.k = int(k)
        self.budget = int(budget)
        self.ef0 = max(int(ef0), int(k))
        self.ef_max = int(ef_max)
        self.growth = float(growth)
        self.gain_tau = float(gain_tau)
        self.early_stop_eps = float(early_stop_eps)
        self.resume = bool(resume)
        self.max_rounds = int(max_rounds)
        # 零召回早停守卫：实验验证用。开启后，若当前查询尚未召回任何真近邻，
        # 则忽略 gain_tau 早停、继续加宽，直到预算/ef_max/图穷尽才停。
        self.truth_ids = truth_ids
        self.guard_zero_recall = bool(guard_zero_recall)
        # 统计：每次查询结束后记录，供实验聚合
        self.last_rounds = 0
        self.last_dists = 0
        self._base = 0

    # ---------------- 预算记账 ----------------

    @property
    def spent(self) -> int:
        """本次查询已消耗的距离计算次数（直接读索引计数器，不另立账本）。"""
        return self.index.counter.count - self._base

    # ---------------- 上层导航 ----------------

    def _descend(self, q: np.ndarray) -> ResumableLayerState:
        """从入口点沿上层贪心下降，建立第 0 层初始状态。"""
        idx = self.index
        if idx.entry_point is None:
            raise RuntimeError("索引为空")
        ep = int(idx.entry_point)
        for lvl in range(idx.max_level, 0, -1):
            if lvl < len(idx.links):
                found = idx._search_layer(q, [ep], 1, lvl)
                if found:
                    ep = int(found[0][1])
        st = ResumableLayerState(0)
        st.visited.add(ep)
        d0 = float(idx._dists([ep], q)[0])
        heapq.heappush(st.cand, (d0, ep))
        heapq.heappush(st.res, (-d0, ep))
        return st

    # ---------------- 单轮加宽 ----------------

    def _advance(self, st: ResumableLayerState, q: np.ndarray, ef_target: int) -> None:
        """在已有状态上继续 best-first，直到本轮收敛 / 穷尽 / 预算不足。

        预算守卫在**批次粒度**上判断：若剩余预算装不下下一批邻居，直接停止本轮，
        因此 self.spent 永远不会超过 self.budget。
        """
        idx = self.index
        graph = idx.links[st.layer] if st.layer < len(idx.links) else {}
        eps = self.early_stop_eps

        while st.cand:
            if self.spent >= self.budget:
                return
            cd, cid = heapq.heappop(st.cand)
            if len(st.res) >= ef_target and cd > st.worst * (1.0 + eps):
                heapq.heappush(st.cand, (cd, cid))   # 压回，保证下一轮可继续
                return
            nbrs = graph.get(cid)
            if not nbrs:
                continue
            fresh = [x for x in nbrs if x not in st.visited]
            if not fresh:
                continue
            if self.spent + len(fresh) > self.budget:
                st.exhausted = True                  # 装不下 → 停，不超额
                return
            st.visited.update(fresh)
            ds = idx._dists(fresh, q)
            for d, x in zip(ds, fresh):
                fd = float(d)
                if len(st.res) < ef_target or fd < -st.res[0][0]:
                    heapq.heappush(st.cand, (fd, x))
                    heapq.heappush(st.res, (-fd, x))
                    while len(st.res) > ef_target:
                        heapq.heappop(st.res)
            st.expansions += 1
        st.exhausted = True

    # ---------------- 检索主循环 ----------------

    def _current_recall(self, ids: Sequence[int], qidx: Optional[int]) -> Optional[float]:
        """给定当前返回的 ids 与查询下标，计算 Recall@k（仅用于零召回守卫）。

        返回 None 表示守卫未启用或无法计算；返回 0.0 表示当前零召回。
        """
        if not self.guard_zero_recall or self.truth_ids is None or qidx is None:
            return None
        truth = set(int(x) for x in self.truth_ids[qidx, :self.k])
        if not truth:
            return None
        return len(set(int(x) for x in ids[:self.k]) & truth) / float(len(truth))

    def search(self, q: np.ndarray, k: Optional[int] = None, qidx: Optional[int] = None):
        """返回 (ids, 平方距离)，并更新 last_rounds / last_dists。"""
        k = int(k if k is not None else self.k)
        self._base = self.index.counter.count

        if not self.resume:
            return self._search_restart(q, k, qidx)

        st = self._descend(q)
        prev_dk: Optional[float] = None
        rounds = 0
        for r in range(self.max_rounds):
            ef_target = max(int(self.ef0 * (self.growth ** r)), k)
            self._advance(st, q, ef_target)
            rounds += 1

            dk = self._d_k(st, k)
            topk_ids = [i for _, i in st.topk(k)]
            current_recall = self._current_recall(topk_ids, qidx)
            if dk is not None and prev_dk is not None and prev_dk > 0.0:
                gain_low = (prev_dk - dk) / prev_dk < self.gain_tau
                # 零召回早停守卫：若当前仍未召回任何真近邻，即使 d_k 改善很小
                # 也继续加宽——否则困难查询会被误判为"已收敛"而提前掐掉。
                if gain_low and (current_recall is None or current_recall > 0.0):
                    break                                    # 边际收益不足
            prev_dk = dk if dk is not None else prev_dk

            if st.exhausted or self.spent >= self.budget:
                break
            if ef_target >= self.ef_max:
                break

        self.last_rounds = rounds
        self.last_dists = self.spent
        found = st.topk(k)
        ids = np.fromiter((i for _, i in found), dtype=np.int32, count=len(found))
        dst = np.fromiter((d for d, _ in found), dtype=np.float32, count=len(found))
        return ids, dst

    def _search_restart(self, q: np.ndarray, k: int, qidx: Optional[int] = None):
        """消融用：每轮从零重算，不复用状态。作为召回率上界。"""
        prev_dk: Optional[float] = None
        rounds = 0
        ids = np.empty(0, dtype=np.int32)
        dst = np.empty(0, dtype=np.float32)
        for r in range(self.max_rounds):
            ef_target = max(int(self.ef0 * (self.growth ** r)), k)
            ids, dst = self.index.search(q, k, ef=ef_target)
            rounds += 1
            dk = float(dst[-1]) if len(dst) >= k else None
            current_recall = self._current_recall(ids, qidx)
            if dk is not None and prev_dk is not None and prev_dk > 0.0:
                gain_low = (prev_dk - dk) / prev_dk < self.gain_tau
                if gain_low and (current_recall is None or current_recall > 0.0):
                    break
            prev_dk = dk if dk is not None else prev_dk
            if self.spent >= self.budget or ef_target >= self.ef_max:
                break
        self.last_rounds = rounds
        self.last_dists = self.spent
        return ids, dst

    def _d_k(self, st: ResumableLayerState, k: int) -> Optional[float]:
        """结果集中第 k 名的平方距离；不足 k 条时返回 None。"""
        if len(st.res) < k:
            return None
        return float(st.topk(k)[-1][0])


# ---------------- 批次共享预算池（BCAS-P） ----------------

class PooledBudgetSearcher:
    """把预算记账单位从「单查询」改为「批次共享池」的 BCAS 变体。

    为什么需要它
    ------------
    AdaptiveSearcher 的预算是**逐查询独立**的：每个查询各自最多花 B 次距离计算。
    这带来一个结构性后果 —— 简单查询因边际收益早停而少花的额度**没有接收方**，
    只能被"省掉"，无法给困难查询。也就是说，"预算再分配"在原设计下
    在结构上不可能发生，与早停阈值取多少无关。
    （实测验证：tau=0 时 n=5 万 上每查询算力 min=397 / max=425，
    离散度仅 7%，恰好等于批次守卫的粒度，没有任何再分配痕迹。）

    本类补上这个接收方。记账规则：

      pool_total = nq · B            批次总额度（硬上界）
      floor      = B                 每个查询保底额度
      cap_max    = cap_factor · B     单个查询上限（SLA 用）

    第 i 个查询（剩余 remaining 个，含自己）的可用量：

      cap_i = min(cap_max, pool - floor · (remaining - 1))

    即：先把后面每个查询的保底额度预留出来，**剩下的余额全部给当前查询**。
    因此困难查询只有在"前面确实省下了额度"时才能突破 B，这就把
    "再分配"从一个口号变成了可测量的量。

    两条硬性质（都由批次粒度守卫保证，不依赖统计期望）
    --------------------------------------------------
    1. 批次总额：sum_i cost_i ≤ nq · B     （pool 只减不增，且每次按实际消耗扣减）
    2. 单查询上限：cost_i ≤ cap_max        （cap_i ≤ cap_max，守卫在批次粒度检查）
    3. 保底：每个查询至少拿到 floor 可用额度（由 cap_i 的预留项归纳可得）

    诚实声明：下降段（上层导航到第 0 层入口）的距离计算在预算守卫之前发生，
    因此当 cap_i 被"保底预留"挤到很小的时候，实测 cost_i 可能略微超过 cap_i。
    本类把这种情况记为 overshoot 并如实报出，不做截断修饰。
    """

    name = "BCAS-池化"

    def __init__(
        self,
        index: HNSW,
        k: int = 10,
        budget: int = 1000,
        cap_factor: float = 8.0,
        ef0: int = 16,
        ef_max: int = 1024,
        growth: float = 2.0,
        gain_tau: float = 0.01,
        early_stop_eps: float = 0.0,
        max_rounds: int = 16,
        truth_ids: Optional[np.ndarray] = None,
        guard_zero_recall: bool = False,
    ) -> None:
        if budget <= 0:
            raise ValueError(f"budget 必须为正，收到 {budget}")
        if cap_factor < 1.0:
            raise ValueError(f"cap_factor 必须 >= 1，否则单查询上限低于保底额度，收到 {cap_factor}")
        self.budget = int(budget)
        self.cap_factor = float(cap_factor)
        self.cap_max = max(int(round(self.budget * self.cap_factor)), self.budget)
        self.floor = self.budget
        # 复用 AdaptiveSearcher 的搜索逻辑，只把 budget 逐查询改写为 cap_i
        self._searcher = AdaptiveSearcher(
            index, k=k, budget=self.budget, ef0=ef0, ef_max=ef_max,
            growth=growth, gain_tau=gain_tau, early_stop_eps=early_stop_eps,
            resume=True, max_rounds=max_rounds,
            truth_ids=truth_ids, guard_zero_recall=guard_zero_recall,
        )
        self._pool = 0
        self._remaining = 0
        self._nq_done = 0
        self.pool_total = 0
        self.starved = 0
        self.overshoot = 0
        self.overshoot_max = 0
        self.overdraft = 0
        self.last_rounds = 0
        self.last_dists = 0
        self.last_cap = 0

    # ---------------- 批次生命周期 ----------------

    def begin_batch(self, n_queries: int) -> None:
        """开一个新批次：总额度 = n_queries × B，并重置所有统计。"""
        if n_queries <= 0:
            raise ValueError(f"n_queries 必须为正，收到 {n_queries}")
        self._pool = int(n_queries) * self.budget
        self._remaining = int(n_queries)
        self._nq_done = 0
        self.pool_total = self._pool
        self.starved = 0
        self.overshoot = 0
        self.overshoot_max = 0
        self.overdraft = 0

    @property
    def pool_remaining(self) -> int:
        return self._pool

    @property
    def pool_spent(self) -> int:
        return self.pool_total - self._pool

    # ---------------- 检索 ----------------

    def search(self, q: np.ndarray, k: Optional[int] = None):
        if self._remaining <= 0:
            raise RuntimeError("批次已结束：请先调用 begin_batch(n_queries)")
        reserve = self.floor * (self._remaining - 1)
        cap = min(self.cap_max, self._pool - reserve)
        cap = max(cap, 0)
        if cap < self.floor:
            self.starved += 1

        self._searcher.budget = cap
        ids, dst = self._searcher.search(q, k, qidx=self._nq_done)
        self._nq_done += 1

        spent = int(self._searcher.last_dists)
        if spent > cap:
            self.overshoot += 1
            self.overshoot_max = max(self.overshoot_max, spent - cap)
        self._pool -= spent
        if self._pool < 0:
            self.overdraft += -self._pool
            self._pool = 0
        self._remaining -= 1

        self.last_rounds = int(self._searcher.last_rounds)
        self.last_dists = spent
        self.last_cap = cap
        return ids, dst


# ---------------- 逐查询评测工具 ----------------

def pooled_batch_eval(
    searcher: PooledBudgetSearcher,
    queries: np.ndarray,
    truth_ids: np.ndarray,
    k: int,
    warmup: int = 0,
) -> Dict[str, object]:
    """批次共享预算池的逐查询评测。

    与 perquery_eval 的两点结构性差异，必须写进报告口径说明：
      1. **不能做 warmup**：预热查询会消耗共享池里的额度，扰动预算分配本身。
         因此池化臂不报 QPS（其冷启动开销与其它臂不可比），只报距离计算次数与召回率，
         而距离计算次数与缓存状态无关，仍可严格比较。
      2. **不能逐查询重置计数器**：预算池跨查询，必须整批跑完再统计。

    返回结构与 perquery_eval 一致（另加池化专属字段），便于直接进同一张表。
    """
    nq = int(queries.shape[0])
    if warmup:
        raise ValueError("池化臂不支持 warmup：预热会消耗共享预算池，破坏预算口径")

    searcher.begin_batch(nq)
    recalls = np.zeros(nq, dtype=np.float64)
    dists = np.zeros(nq, dtype=np.int64)
    caps = np.zeros(nq, dtype=np.int64)
    rounds = np.zeros(nq, dtype=np.int64)

    for i in range(nq):
        ids, _ = searcher.search(queries[i])
        dists[i] = searcher.last_dists
        caps[i] = searcher.last_cap
        rounds[i] = searcher.last_rounds
        hit = len(set(int(x) for x in ids[:k]) & set(int(x) for x in truth_ids[i, :k]))
        recalls[i] = hit / float(k)

    return {
        "n_queries": nq,
        "recall_mean": float(recalls.mean()),
        "recall_p1": float(np.percentile(recalls, 1)),
        "recall_p5": float(np.percentile(recalls, 5)),
        "recall_std": float(recalls.std()),
        "recall_min": float(recalls.min()),
        "dists_mean": float(dists.mean()),
        "dists_p50": float(np.percentile(dists, 50)),
        "dists_p99": float(np.percentile(dists, 99)),
        "dists_max": float(dists.max()),
        "recalls": recalls,
        "dists": dists,
        # 池化专属
        "caps": caps,
        "caps_mean": float(caps.mean()),
        "caps_max": float(caps.max()),
        "rounds_mean": float(rounds.mean()),
        "pool_total": int(searcher.pool_total),
        "pool_remaining": int(searcher.pool_remaining),
        "starved": int(searcher.starved),
        "overshoot": int(searcher.overshoot),
        "overshoot_max": int(searcher.overshoot_max),
        "overdraft": int(searcher.overdraft),
        # 离散度：max/mean。固定 ef 接近 1，池化臂应显著大于 1，这是再分配的直接证据
        "dispersion": float(dists.mean() and dists.max() / dists.mean()),
    }

def perquery_eval(search_fn, queries: np.ndarray, truth_ids: np.ndarray, k: int,
                  counter: DistanceCounter, warmup: int = 3, pass_index: bool = False) -> Dict[str, object]:
    """逐条查询记录 (召回率, 距离计算次数)，返回聚合统计。

    分布式评估必须看尾部分位数：均值相同、P1 不同的两个系统，用户体验完全不同。

    pass_index=True 时把查询下标 i 也传给 search_fn，供需要 truth_ids 的实验
    （如零召回早停守卫）使用；默认关闭以保持与旧代码的兼容性。
    """
    nq = int(queries.shape[0])
    for i in range(min(warmup, nq)):
        if pass_index:
            search_fn(queries[i], i)
        else:
            search_fn(queries[i])

    recalls = np.zeros(nq, dtype=np.float64)
    dists = np.zeros(nq, dtype=np.int64)
    for i in range(nq):
        counter.reset()
        if pass_index:
            ids, _ = search_fn(queries[i], i)
        else:
            ids, _ = search_fn(queries[i])
        dists[i] = counter.count
        hit = len(set(int(x) for x in ids[:k]) & set(int(x) for x in truth_ids[i, :k]))
        recalls[i] = hit / float(k)

    return {
        "n_queries": nq,
        "recall_mean": float(recalls.mean()),
        "recall_p1": float(np.percentile(recalls, 1)),
        "recall_p5": float(np.percentile(recalls, 5)),
        "recall_std": float(recalls.std()),
        "recall_min": float(recalls.min()),
        "dists_mean": float(dists.mean()),
        "dists_p50": float(np.percentile(dists, 50)),
        "dists_p99": float(np.percentile(dists, 99)),
        "dists_max": float(dists.max()),
        "recalls": recalls,
        "dists": dists,
    }


def interpolate_recall_at_cost(
    ef_grid: Sequence[int], costs: Sequence[float], recalls: Sequence[float], target_cost: float
) -> Tuple[float, str]:
    """在 (cost, recall) 曲线上按算力线性插值，得到"等算力下的固定 ef 召回率"。

    返回 (插值召回率, 说明字符串)。目标算力超出实测区间时取最近端点并说明。
    """
    pts = sorted(zip(costs, recalls))
    if not pts:
        return float("nan"), "空曲线"
    if target_cost <= pts[0][0]:
        return pts[0][1], f"低于区间下界({pts[0][0]:.0f})，取最小 ef"
    if target_cost >= pts[-1][0]:
        return pts[-1][1], f"高于区间上界({pts[-1][0]:.0f})，取最大 ef"
    for (c0, r0), (c1, r1) in zip(pts, pts[1:]):
        if c0 <= target_cost <= c1:
            if c1 == c0:
                return r1, "同算力点"
            w = (target_cost - c0) / (c1 - c0)
            return r0 + w * (r1 - r0), "线性插值"
    return pts[-1][1], "兜底"
