"""评测口径。

三条指标定义必须写死在报告里，评审会照着复现：
  recall@k   = 逐查询求 |返回集合 ∩ 真值集合| / k，再对查询取平均
  qps        = 查询条数 / 总墙钟时间（含单查询的 Python 调用开销，这是最保守的算法）
  dists/query= 平均每次查询触发的**距离计算次数** —— 与硬件无关的纯算法层指标
"""

from __future__ import annotations

import numpy as np


def recall_at_k(approx: np.ndarray, truth: np.ndarray, k: int) -> float:
    """平均 Recall@k。approx 中不足 k 的位置用 -1 填充，不会误命中。"""
    kk = int(min(k, truth.shape[1]))
    if kk <= 0:
        return 0.0
    a = approx[:, :kk]
    t = truth[:, :kk]
    hits = 0
    for i in range(a.shape[0]):
        hits += len(set(a[i].tolist()) & set(t[i].tolist()))
    return hits / (a.shape[0] * kk)


def recall_at_k_fast(approx: np.ndarray, truth: np.ndarray, k: int) -> float:
    """向量化版本，逻辑等价，大实验用。"""
    kk = int(min(k, truth.shape[1]))
    a = approx[:, :kk]
    t = truth[:, :kk]
    hit = (a[:, :, None] == t[:, None, :]).any(axis=2)
    return float(hit.mean())


def qps(n_queries: int, elapsed_s: float) -> float:
    return n_queries / elapsed_s if elapsed_s > 0 else float("inf")


def dists_per_query(total_dists: int, n_queries: int) -> float:
    return total_dists / n_queries if n_queries else 0.0


def speedup_vs(numerator: float, baseline: float) -> float:
    """baseline / numerator，例如 QPS 提升倍数。"""
    return baseline / numerator if numerator > 0 else float("inf")


def format_table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 4) -> str:
    """把 list[dict] 渲染成 markdown 表，便于直接贴进技术报告。"""
    head = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for _, key in columns:
            v = r.get(key)
            if isinstance(v, float):
                cells.append(f"{v:.{digits}f}")
            elif v is None:
                cells.append("-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
