"""冒烟测试：30 秒内跑通全链路，确认每个模块都能出数字。

用途：环境装好后先跑这个。它过了，说明数据、构图、搜索、计数、评测五个环节都正常，
后面跑正式实验只是把 n 调大而已。

    python experiments/exp0_smoke.py
"""

from __future__ import annotations

import time

import numpy as np

from _common import banner
from src.baselines import BruteForceIndex, make_baselines
from src.bench import benchmark_search
from src.data import make_synthetic, brute_force_knn
from src.hnsw import HNSW


def main() -> None:
    banner("1/5 生成合成数据（带簇结构，自带 ground truth）")
    ds = make_synthetic(n=3000, dim=32, nq=100, k=10, seed=7)
    print(ds.describe())

    banner("2/5 校验真值正确性 —— 真值不能错，错了后面全白做")
    ids, _ = brute_force_knn(ds.train, ds.queries[:5], ds.k)
    same = (ids == ds.neighbors[:5]).all()
    print("暴力搜索与数据集自带真值一致：", "是" if same else "否（有问题！）")
    assert same, "ground truth 校验失败"

    banner("3/5 构建自实现 HNSW")
    t0 = time.perf_counter()
    idx = HNSW(dim=ds.dim, m=8, ef_construction=40, ef_search=32, seed=0)
    idx.add_items(ds.train, verbose_every=1000)
    build_s = time.perf_counter() - t0
    print(f"构图耗时 {build_s:.2f}s")
    print(idx.memory_report())
    build_dists = idx.counter.count
    print(f"构图距离计算 {build_dists:,}")

    banner("4/5 检索：召回率必须随 ef 单调上升，否则是 bug")
    prev = -1.0
    for ef in (4, 8, 16, 32, 64, 128):
        r = benchmark_search(idx, ds.queries, ds.neighbors, ds.k, ef=ef, warmup=2)
        print(
            f"ef={ef:<4d} Recall@{ds.k}={r.recall:.4f}  QPS={r.qps:9.1f}  "
            f"延迟={r.latency_ms:6.3f}ms  距离计算/查询={r.dists_per_query:8.1f}"
        )
        assert r.recall >= prev - 1e-9, f"召回率在 ef={ef} 处下降，构图或搜索可疑"
        prev = r.recall

    banner("5/5 基线对照")
    bf = BruteForceIndex().add_items(ds.train)
    r = benchmark_search(bf, ds.queries, ds.neighbors, ds.k, warmup=1)
    print(
        f"BruteForce  Recall@{ds.k}={r.recall:.4f}  QPS={r.qps:9.1f}  "
        f"距离计算/查询={r.dists_per_query:8.1f}"
    )
    assert r.recall > 0.9999, "暴力搜索召回率应为 1.0"

    print("\n可选基线：", ", ".join(make_baselines(ds.dim).keys()))
    print("\n冒烟通过。把 n 调到 10 万、换成 SIFT1M 即可跑正式实验。")


if __name__ == "__main__":
    main()
