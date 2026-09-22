"""主实验：Recall@k – QPS 前沿（报告第 4 章的核心表）。

    python experiments/exp1_recall_qps.py --synthetic --n 30000           # 无下载快速版
    python experiments/exp1_recall_qps.py --n 100000 --queries 1000       # SIFT1M 子集

产出：results/exp1_recall_qps.csv / .json，以及可直接贴进报告的 markdown 表。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _common import banner, build_or_load_index, save_rows
from src.baselines import BruteForceIndex, make_baselines
from src.bench import benchmark_search, pareto_frontier
from src.data import Dataset, load_hdf5, make_synthetic
from src.metrics import format_table


def load_dataset(args: argparse.Namespace) -> Dataset:
    banner("载入数据")
    if args.synthetic:
        ds = make_synthetic(n=args.n, dim=64, nq=args.queries, k=args.k)
    else:
        ds = load_hdf5(args.dataset)
        if args.n < ds.n:
            banner(f"底库子采样至 {args.n}，重算 ground truth（这一步必须做）")
            t0 = time.perf_counter()
            ds = ds.with_train_subset(args.n, recompute_truth=True)
            print(f"完成，耗时 {time.perf_counter() - t0:.1f}s")
        if args.queries < ds.nq:
            ds = Dataset(
                f"{ds.name}|q={args.queries}",
                ds.train,
                np.ascontiguousarray(ds.queries[: args.queries]),
                np.ascontiguousarray(ds.neighbors[: args.queries]),
                np.ascontiguousarray(ds.distances[: args.queries]),
            )
    print(ds.describe())
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--synthetic", action="store_true", help="不下载，用合成数据")
    ap.add_argument("--n", type=int, default=100000, help="底库规模（小于数据集时自动子采样）")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=100, help="ef_construction")
    ap.add_argument("--bf-queries", type=int, default=200, help="暴力基线只跑这么多查询")
    ap.add_argument("--reuse-index", action="store_true", default=True,
                    help="复用 data/index_cache 里已构好的图（默认开）")
    ap.add_argument("--no-reuse-index", dest="reuse_index", action="store_false")
    ap.add_argument("--tag", default="exp1_recall_qps")
    args = ap.parse_args()

    ds = load_dataset(args)
    rows: list[dict] = []

    banner("构建自实现 HNSW（本项目方法）")
    ours, ours_build = build_or_load_index(
        ds.train, ds.dim, ds.name, m=args.m, efc=args.efc, seed=0,
        reuse=args.reuse_index, verbose_every=max(1, ds.n // 10),
    )
    # 缓存命中时 counter 里存着构图期的计数（随索引一起被 pickle 下来），所以下面这句两路都对
    ours_build_dists = ours.counter.count
    build_txt = f"{ours_build:.1f}s" if ours_build is not None else "未知（缓存无元数据）"
    print(f"构建耗时 {build_txt} | 构图距离计算 {ours_build_dists:,}")
    print(ours.memory_report())

    for ef in args.ef:
        r = benchmark_search(ours, ds.queries, ds.neighbors, args.k, ef=ef)
        r.build_s = ours_build
        row = r.as_row()
        row["方法"] = "HNSW(ours)"
        rows.append(row)
        print(f"ef={ef:<4d} Recall={r.recall:.4f} QPS={r.qps:8.1f} 距离/查询={r.dists_per_query:8.1f}")

    banner("基线")
    baselines = make_baselines(ds.dim, m=args.m, ef_construction=args.efc)
    for name, base in baselines.items():
        t0 = time.perf_counter()
        base.add_items(ds.train)
        bt = time.perf_counter() - t0
        if name == "BruteForce":
            r = benchmark_search(base, ds.queries, ds.neighbors, args.k, warmup=1,
                                 limit_queries=args.bf_queries)
            r.build_s = bt
            rows.append(r.as_row())
            print(f"{name:<14s} Recall={r.recall:.4f} QPS={r.qps:8.1f} 距离/查询={r.dists_per_query:8.1f}")
            continue
        for ef in args.ef:
            r = benchmark_search(base, ds.queries, ds.neighbors, args.k, ef=ef)
            r.build_s = bt
            row = r.as_row()
            row["方法"] = name
            rows.append(row)
            print(f"{name:<14s} ef={ef:<4d} Recall={r.recall:.4f} QPS={r.qps:8.1f}")

    banner("结果表（可直接贴报告）")
    cols = [
        ("方法", "方法"), ("ef", "ef"), ("Recall@k", "Recall@k"),
        ("QPS", "QPS"), ("延迟(ms)", "延迟(ms)"),
        ("距离计算/次查询", "距离计算/次查询"), ("构建(s)", "构建(s)"),
    ]
    table = format_table(rows, cols, digits=4)
    print(table)

    path = save_rows(args.tag, rows)
    print(f"\n已保存 {path}")
    with open(path.replace(".csv", ".md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {args.tag}\n\n数据：{ds.describe()}\n\n{table}\n")

    banner("自实现方法的 Pareto 前沿")
    ours_rows = [r for r in rows if r["方法"] == "HNSW(ours)"]
    print(format_table(ours_rows, cols, digits=4))


if __name__ == "__main__":
    main()
