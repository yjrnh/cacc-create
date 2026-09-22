"""统一评测：召回率、QPS、距离计算次数。

计时口径刻意选最保守的一种：整个查询循环的墙钟时间，含 Python 单次调用开销。
不做批量化、不做多线程 —— 因为报告要比的是**算法效率**，不是工程优化水平。
距离计算次数则完全不受这套保守口径影响，这也是它为什么值得单独列一栏。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .counter import DistanceCounter
from .metrics import recall_at_k_fast


@dataclass
class BenchResult:
    name: str
    recall: float
    qps: float
    latency_ms: float
    dists_per_query: float
    n_queries: int
    ef: Optional[int] = None
    build_s: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "方法": self.name,
            "ef": self.ef if self.ef is not None else "-",
            "Recall@k": self.recall,
            "QPS": self.qps,
            "延迟(ms)": self.latency_ms,
            "距离计算/次查询": self.dists_per_query,
            "构建(s)": self.build_s,
        }
        row.update(self.extra)
        return row


def benchmark_search(
    index,
    queries: np.ndarray,
    truth_ids: np.ndarray,
    k: int,
    ef: Optional[int] = None,
    warmup: int = 5,
    limit_queries: Optional[int] = None,
    counter: Optional[DistanceCounter] = None,
) -> BenchResult:
    """在给定 ef 下跑完整查询集，返回一条结果。

    limit_queries 用于暴力基线这种"每条查询都扫全库"的慢速对照：只跑前 N 条。
    **真值必须跟着一起截断** —— 预测 200 条、真值 500 行会在广播时炸出一个
    与真实原因无关的形状错误（曾经真炸过）。
    """
    qs = queries if limit_queries is None else queries[:limit_queries]
    ts = truth_ids if limit_queries is None else truth_ids[:limit_queries]
    nq = int(qs.shape[0])
    if int(ts.shape[0]) != nq:
        raise ValueError(f"查询数 {nq} 与真值行数 {ts.shape[0]} 不一致；"
                         "limit_queries 必须同时作用于两者")
    ctr = counter if counter is not None else getattr(index, "counter", None)
    if not getattr(index, "counts_distances", True):
        ctr = None

    for i in range(min(warmup, nq)):                      # 预热，排除首次分配的抖动
        index.search(qs[i], k, ef=ef)

    approx = np.full((nq, k), -1, dtype=np.int32)
    if ctr is not None:
        ctr.reset()
    t0 = time.perf_counter()
    for i in range(nq):
        ids, _ = index.search(qs[i], k, ef=ef)
        m = min(len(ids), k)
        approx[i, :m] = ids[:m]
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rec = recall_at_k_fast(approx, ts, k)
    total_dists = ctr.count if ctr is not None else -1
    return BenchResult(
        name=getattr(index, "name", type(index).__name__),
        recall=rec,
        qps=nq / elapsed if elapsed > 0 else float("inf"),
        latency_ms=1e3 * elapsed / nq,
        dists_per_query=(total_dists / nq) if total_dists >= 0 else float("nan"),
        n_queries=nq,
        ef=ef,
        extra={"总距离计算": float(total_dists)} if total_dists >= 0 else {},
    )


def sweep_ef(
    index,
    queries: np.ndarray,
    truth_ids: np.ndarray,
    k: int,
    ef_values: List[int],
    **kw,
) -> List[BenchResult]:
    return [benchmark_search(index, queries, truth_ids, k, ef=ef, **kw) for ef in ef_values]


def pareto_frontier(results: List[BenchResult]) -> List[BenchResult]:
    """按 QPS 降序保留非支配点（召回更高且 QPS 更高才算支配）。"""
    ordered = sorted(results, key=lambda r: r.qps, reverse=True)
    front: List[BenchResult] = []
    best_recall = -1.0
    for r in ordered:
        if r.recall > best_recall:
            front.append(r)
            best_recall = r.recall
    return front
