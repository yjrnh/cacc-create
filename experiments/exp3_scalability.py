"""规模实验：底库从 1 万到 10 万+，看构图代价与检索代价怎么长（报告第 5 章）。

    python experiments/exp3_scalability.py --synthetic --sizes 5000 10000 20000
    python experiments/exp3_scalability.py --sizes 10000 30000 100000 --queries 500

记录：构图耗时、构图距离计算、边数、平均出度、Recall、QPS、每次查询距离计算。
关键看图：**每次查询的距离计算应随 log n 增长**（图索引的根本优势），
如果它接近线性增长，说明图的可导航性没建起来。
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import banner, build_or_load_index, save_rows
from src.bench import benchmark_search
from src.data import Dataset, load_hdf5, make_synthetic
from src.metrics import format_table


def base_dataset(args: argparse.Namespace) -> Dataset:
    if args.synthetic:
        return make_synthetic(n=max(args.sizes), dim=64, nq=args.queries, k=args.k)
    return load_hdf5(args.dataset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 30000, 100000])
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=80)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=100)
    args = ap.parse_args()

    full = base_dataset(args)
    rows: list[dict] = []

    for n in sorted(args.sizes):
        banner(f"n = {n:,}")
        if n <= full.n:
            ds = full if n == full.n else full.with_train_subset(n, recompute_truth=True)
        else:
            raise SystemExit(f"请求的 n={n} 超过数据集规模 {full.n}")
        if args.queries < ds.nq:
            ds = Dataset(
                ds.name, ds.train,
                np.ascontiguousarray(ds.queries[: args.queries]),
                np.ascontiguousarray(ds.neighbors[: args.queries]),
                np.ascontiguousarray(ds.distances[: args.queries]),
            )

        # 走公共缓存：n=2万 这张图同时是 exp2 / exp4 的底图，构一次全实验共用。
        idx, build_s = build_or_load_index(
            ds.train, ds.dim, ds.name, m=args.m, efc=args.efc, seed=0,
            verbose_every=max(1, n // 5), early_stop_eps=None,
        )
        build_dists = idx.counter.count
        st = idx.edge_stats()
        r = benchmark_search(idx, ds.queries, ds.neighbors, args.k, ef=args.ef)
        if build_s is None:
            # 只在复用了没有 sidecar 的历史缓存时发生。不谎报为 0，如实留空。
            print("  [warn] 该缓存无构建耗时元数据，表格里记为空", flush=True)

        rows.append(
            {
                "n": int(n),
                "构图耗时(s)": build_s,
                "构图距离计算": float(build_dists),
                "构图距离/n": build_dists / n,
                "边数": float(st["edges"]),
                "平均出度": float(st["avg_degree"]),
                "图内存(C++布局,MB)": st["graph_bytes_if_cpp"] / 2**20,
                "Recall@k": r.recall,
                "QPS": r.qps,
                "距离计算/次查询": r.dists_per_query,
                "距离比n": r.dists_per_query / n,
            }
        )
        shown = f"{build_s:7.1f}s" if build_s is not None else "  复用(无记录)"
        print(
            f"构图 {shown} | 出度 {st['avg_degree']:5.2f} | "
            f"Recall {r.recall:.4f} | QPS {r.qps:8.1f} | 距离/查询 {r.dists_per_query:8.1f} "
            f"({r.dists_per_query / n * 100:.3f}% of n)",
            flush=True,
        )

    cols = [
        ("n", "n"), ("构图耗时(s)", "构图耗时(s)"), ("构图距离计算", "构图距离计算"),
        ("构图距离/n", "构图距离/n"), ("边数", "边数"), ("平均出度", "平均出度"),
        ("图内存(C++布局,MB)", "图内存(C++布局,MB)"), ("Recall@k", "Recall@k"),
        ("QPS", "QPS"), ("距离计算/次查询", "距离计算/次查询"), ("距离比n", "距离比n"),
    ]
    table = format_table(rows, cols, digits=4)
    print("\n" + table)
    path = save_rows("exp3_scalability", rows)
    with open("results/exp3_scalability.md", "w", encoding="utf-8") as fh:
        fh.write(f"# 规模实验\n\nm={args.m} ef_construction={args.efc} ef={args.ef}\n\n{table}\n")
    print(f"\n已保存 {path}")


if __name__ == "__main__":
    main()
