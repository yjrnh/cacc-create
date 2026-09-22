"""自实现 HNSW 近邻图索引 —— 本项目的算法核心。

与 Faiss / hnswlib 的差别（也就是报告里的增量位置）：

1. **邻居选择启发式可插拔**
   simple    = 论文 Algorithm 3，直接取最近的 m 个邻居；
   heuristic = 论文 Algorithm 4，候选 e 只有在「离 q 比离已选集合 R 中任何元素都近」
               时才被保留 —— 直觉是被近邻"遮挡"的候选属于冗余连接，占出度不贡献可达性。
   再额外暴露 extend_candidates / keep_pruned 两个开关，构成 2×2 消融。

2. **(1+eps) 近似提前终止**
   搜索循环的停机条件从 `候选距离 > 结果集最差距离` 放宽为
   `候选距离 > (1+eps) × 结果集最差距离`，用一点召回率换 QPS。
   注意：本实现内部所有距离都是**平方欧氏距离**，因此 eps 作用在平方距离上，
   与论文里定义在距离上的 eps 不完全等价 —— 报告里必须写清这一点。

3. **可计量**
   所有距离计算经 DistanceCounter 计数，可给出与硬件无关的算法层指标。
   Faiss / hnswlib 均不对外暴露距离计算次数，这是本项目选择自实现的原因之一。
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .counter import DistanceCounter


class HNSW:
    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        neighbor_mode: str = "heuristic",
        extend_candidates: bool = False,
        keep_pruned: bool = True,
        early_stop_eps: float = 0.0,
        level_mult: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        if neighbor_mode not in ("simple", "heuristic"):
            raise ValueError("neighbor_mode 必须是 'simple' 或 'heuristic'")
        self.dim = int(dim)
        self.m = int(m)
        self.m0 = 2 * int(m)          # 第 0 层出度上限为 2m（论文约定）
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.neighbor_mode = neighbor_mode
        self.extend_candidates = bool(extend_candidates)
        self.keep_pruned = bool(keep_pruned)
        self.early_stop_eps = float(early_stop_eps)
        self.level_mult = (1.0 / math.log(self.m)) if level_mult is None else float(level_mult)
        self._rng = np.random.default_rng(seed)

        self.vectors: Optional[np.ndarray] = None
        self.links: List[Dict[int, List[int]]] = []
        self.level_of: List[int] = []
        self.entry_point: Optional[int] = None
        self.max_level: int = -1
        self.counter = DistanceCounter()
        self.name = "HNSW(ours)"

    # ---------------- 基础设施 ----------------

    def _dists(self, ids: Sequence[int], q: np.ndarray) -> np.ndarray:
        """对给定 id 批量算平方欧氏距离，并计数。"""
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size == 0:
            return np.empty(0, dtype=np.float32)
        diff = self.vectors[arr] - q
        d2 = np.einsum("ij,ij->i", diff, diff)
        self.counter.add(int(arr.size))
        return d2

    def _ensure_layer(self, level: int) -> Dict[int, List[int]]:
        while len(self.links) <= level:
            self.links.append({})
        return self.links[level]

    def _sample_level(self) -> int:
        """层高抽样：P(level >= l) = m^-l，等价于 floor(-ln(U) / ln(m))。"""
        u = float(self._rng.random())
        while u <= 0.0:
            u = float(self._rng.random())
        return int(-math.log(u) * self.level_mult)

    # ---------------- 单层搜索 ----------------

    def _search_layer(
        self, q: np.ndarray, entry_ids: Sequence[int], ef: int, layer: int
    ) -> List[Tuple[float, int]]:
        """单层 best-first 搜索。返回按距离升序的 (平方距离, id)，长度 <= ef。

        ef=1 时退化为论文里的贪心下降（用于上层导航）。

        early_stop_eps 的方向性（**很容易搞反，这里写死**）
        ------------------------------------------------
        终止判据是 `候选距离 > 当前第 ef 名的距离 × (1 + eps)`，eps=0 即论文原始形式。
        因此：

          * eps > 0 → 阈值被**放宽** → 更难触发终止 → 搜得更深 → 算力↑、召回↑
            （论文称之为"relaxed stop condition"，relaxed 指的是"放松了停止的要求"，
             **不是**"更早停止"）
          * eps < 0 → 阈值被**收紧** → 更早终止 → 算力↓、召回↓（真正的省算力方向）
          * eps <= -1 → 阈值退化为 0，第一轮即终止，无意义，直接报错

        参数名保留了历史叫法 `early_stop_eps`，但**它本身不是"越大越早停"**。
        要用它省算力，必须取负值。见 experiments/exp2_ablation.py 的双向扫描。
        """
        if self.early_stop_eps <= -1.0:
            raise ValueError(f"early_stop_eps={self.early_stop_eps} 会使终止阈值退化为 0，"
                             "搜索在第一轮就结束；合法取值应 > -1")
        ef = max(int(ef), 1)
        ents = [int(x) for x in entry_ids]
        visited = set(ents)
        cand: List[Tuple[float, int]] = []   # 最小堆：待扩展
        res: List[Tuple[float, int]] = []    # 最大堆（存负值）：当前最优 ef 个

        d0 = self._dists(ents, q)
        for d, i in zip(d0, ents):
            fd = float(d)
            heapq.heappush(cand, (fd, i))
            heapq.heappush(res, (-fd, i))

        graph = self.links[layer] if layer < len(self.links) else {}

        while cand:
            cd, cid = heapq.heappop(cand)
            if len(res) >= ef:
                worst = -res[0][0]
                if cd > worst * (1.0 + self.early_stop_eps):
                    break
            nbrs = graph.get(cid)
            if not nbrs:
                continue
            fresh = [x for x in nbrs if x not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            ds = self._dists(fresh, q)
            for d, x in zip(ds, fresh):
                fd = float(d)
                if len(res) < ef or fd < -res[0][0]:
                    heapq.heappush(cand, (fd, x))
                    heapq.heappush(res, (-fd, x))
                    if len(res) > ef:
                        heapq.heappop(res)

        return sorted((-nd, i) for nd, i in res)

    # ---------------- 邻居选择（消融主战场） ----------------

    def _select_neighbors(
        self, base_id: int, cand_ids: Sequence[int], limit: int, layer: int
    ) -> List[int]:
        """HNSW 论文 Algorithm 3 / Algorithm 4。"""
        if not cand_ids:
            return []
        cand = [int(c) for c in cand_ids]
        ds = self._dists(cand, self.vectors[base_id])
        order = np.argsort(ds, kind="stable")

        if self.neighbor_mode == "simple":
            return [cand[i] for i in order[:limit]]

        if self.extend_candidates and layer < len(self.links):
            graph = self.links[layer]
            base = set(cand)
            extra = set()
            for c in cand:
                for nb in graph.get(c, ()):
                    nb = int(nb)
                    if nb not in base and nb != base_id:
                        extra.add(nb)
            if extra:
                extra_ids = sorted(extra)
                d_extra = self._dists(extra_ids, self.vectors[base_id])
                cand = cand + extra_ids
                ds = np.concatenate([ds, d_extra])
                order = np.argsort(ds, kind="stable")

        selected: List[int] = []
        discarded: List[int] = []
        for i in order:
            if len(selected) >= limit:
                break
            c = cand[i]
            if selected:
                d_sel = self._dists(selected, self.vectors[c])
                if float(d_sel.min()) < float(ds[i]):
                    discarded.append(c)
                    continue
            selected.append(c)
        if self.keep_pruned and len(selected) < limit:
            for c in discarded:
                if len(selected) >= limit:
                    break
                selected.append(c)
        return selected

    # ---------------- 插入 ----------------

    def _insert(self, idx: int, vec: np.ndarray) -> None:
        lvl = self._sample_level()
        self.level_of.append(lvl)

        if self.entry_point is None:
            for l in range(lvl + 1):
                self._ensure_layer(l)[idx] = []
            self.entry_point = idx
            self.max_level = lvl
            return

        eps = [self.entry_point]
        for l in range(self.max_level, lvl, -1):      # 上层贪心下降
            if l < len(self.links):
                eps = [x[1] for x in self._search_layer(vec, eps, 1, l)]

        for l in range(min(lvl, self.max_level), -1, -1):
            graph = self._ensure_layer(l)
            found = self._search_layer(vec, eps, self.ef_construction, l)
            cand = [x[1] for x in found]
            limit = self.m0 if l == 0 else self.m
            sel = self._select_neighbors(idx, cand, limit, l)
            graph.setdefault(idx, [])
            graph[idx] = list(sel)
            for nb in sel:                            # 反向连接 + 超度剪枝
                lst = graph.setdefault(nb, [])
                if idx not in lst:
                    lst.append(idx)
                if len(lst) > limit:
                    graph[nb] = list(self._select_neighbors(nb, list(lst), limit, l))
            if cand:
                eps = cand

        for l in range(0, lvl + 1):                   # 保证节点在所有层有记录
            self._ensure_layer(l).setdefault(idx, [])

        if lvl > self.max_level:
            self.max_level = lvl
            self.entry_point = idx

    def add_items(self, vectors: np.ndarray, verbose_every: int = 0) -> "HNSW":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.vectors = vectors
        self.dim = int(vectors.shape[1])
        n = vectors.shape[0]
        for i in range(n):
            self._insert(i, vectors[i])
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"  已插入 {i + 1}/{n}  距离计算 {self.counter.count:,}", flush=True)
        return self

    # ---------------- 检索 ----------------

    def search(self, q: np.ndarray, k: int, ef: Optional[int] = None):
        if self.entry_point is None:
            raise RuntimeError("索引为空")
        ef = max(int(ef if ef is not None else self.ef_search), int(k))
        eps = [self.entry_point]
        for l in range(self.max_level, 0, -1):
            if l < len(self.links):
                eps = [x[1] for x in self._search_layer(q, eps, 1, l)]
        found = self._search_layer(q, eps, ef, 0)[:k]
        ids = np.fromiter((x[1] for x in found), dtype=np.int32, count=len(found))
        dst = np.fromiter((x[0] for x in found), dtype=np.float32, count=len(found))
        return ids, dst

    # ---------------- 统计 ----------------

    def edge_stats(self) -> Dict[str, float]:
        """图结构统计。边数 × 4 字节即为 C++ 布局下的图内存，与语言无关。"""
        edges = 0
        for level, graph in enumerate(self.links):
            for lst in graph.values():
                edges += len(lst)
        nodes = sum(len(g) for g in self.links)
        return {
            "levels": len(self.links),
            "nodes_total": nodes,
            "edges": edges,
            "avg_degree": edges / nodes if nodes else 0.0,
            "graph_bytes_if_cpp": 4 * edges,
        }

    def memory_report(self) -> str:
        st = self.edge_stats()
        graph_cpp = st["graph_bytes_if_cpp"] / 2**20
        vec_mb = (self.vectors.nbytes / 2**20) if self.vectors is not None else 0.0
        return (
            f"层数={st['levels']} 边数={int(st['edges']):,} 平均出度={st['avg_degree']:.2f} | "
            f"图内存(C++布局)={graph_cpp:.2f}MB 向量={vec_mb:.1f}MB"
        )

    # ---------------- 持久化（构图慢，实验之间不要重复构） ----------------

    def save(self, path: str) -> str:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=5)
        return path

    @staticmethod
    def load(path: str) -> "HNSW":
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)
