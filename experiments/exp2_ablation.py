"""消融实验：邻居选择策略 + (1+eps) 终止阈值（报告 4.5 节的消融表）。

    python experiments/exp2_ablation.py --synthetic --n 20000
    python experiments/exp2_ablation.py --n 50000 --queries 500

两组消融：
  A. 邻居选择：simple / heuristic × extend_candidates × keep_pruned，共 5 种配置
  B. 终止阈值：eps ∈ {-0.75, -0.5, -0.25, 0, 0.25, 0.5}（**双向**扫描）

为什么 eps 必须双向扫（这一节是本项目的一个反直觉发现）
------------------------------------------------------
终止判据是 `候选距离 > 当前第 ef 名的距离 × (1 + eps)`。所以：

    eps > 0 → 阈值放宽 → 终止更难触发 → 搜得更深 → 算力**上升**、召回上升
    eps < 0 → 阈值收紧 → 更早终止     → 算力**下降**、召回下降

也就是说，**"eps 越大越省算力"是错的**，方向恰好相反。只扫正值会得到
"召回不掉、算力还涨"的困惑结果（这正是第一版实验踩到的坑）。
想验证提前终止能不能省算力，必须扫负值区间。eps 取 (-1, 0) 是合法且有效的。

注意两点口径：
  * eps 作用在**平方距离**上（实现内不取 sqrt），报告里必须写清；
  * 本组消融**只改查询期的 eps**，5 个 eps 档位共用同一张图（构图后修改实例属性），
    因此组内差异全部来自搜索期的终止策略，不掺构图差异。这是一个必要的控制。

工程说明（为什么这个脚本看起来比别的复杂）
--------------------------------------------
5 种邻居配置意味着要构 **5 张不同的图**，彼此完全独立。串行跑在 n=2 万上要 20 多分钟，
纯属浪费。本脚本用「父进程分发 / 子进程各跑一个配置」的方式并行化：

  * 不用 multiprocessing.Pool —— Windows 下 spawn 要重新 pickle 数据集，
    还有 sys.path 与 __mp_main__ 的坑；直接起子进程最省事也最可移植。
  * 每个子进程**自己重建数据集**（seed=0，逐位一致），所以不需要传大数据。
  * 子进程把结果以标记行打到 stdout，父进程用线程并发 drain（communicate 会释放 GIL）。
    刻意不用临时文件：本机执行沙箱会在删除文件时抛 safe-delete 错误，临时文件的生命周期
    反而是个坑；stdout 少一处状态。
  * 图本身走 `data/index_cache/` 缓存，重跑同一个配置几乎零成本。

消融 B 复用配置 B（heuristic 默认）那张图，所以总共只构 5 张图而不是 6 张。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from _common import RESULTS, banner, build_or_load_index, save_rows
from src.bench import benchmark_search
from src.data import Dataset, load_hdf5, make_synthetic
from src.metrics import format_table

RESULT_MARK = "@@EXP2_RESULT@@"

NEIGHBOR_CONFIGS = [
    ("simple", dict(neighbor_mode="simple")),
    ("heuristic（默认）", dict(neighbor_mode="heuristic")),
    ("heuristic, 不补足剪枝", dict(neighbor_mode="heuristic", keep_pruned=False)),
    ("heuristic, 扩展候选", dict(neighbor_mode="heuristic", extend_candidates=True)),
    (
        "heuristic, 扩展+不补足",
        dict(neighbor_mode="heuristic", extend_candidates=True, keep_pruned=False),
    ),
]
DEFAULT_CONFIG_IDX = 1          # 消融 B 复用这一张图


def load_dataset(args: argparse.Namespace) -> Dataset:
    if args.synthetic:
        return make_synthetic(n=args.n, dim=64, nq=args.queries, k=args.k)
    ds = load_hdf5(args.dataset)
    if args.n < ds.n:
        banner(f"底库子采样至 {args.n}，重算 ground truth（这一步必须做）")
        t0 = time.perf_counter()
        ds = ds.with_train_subset(args.n, recompute_truth=True)
        print(f"完成，耗时 {time.perf_counter() - t0:.1f}s")
    if args.queries < ds.nq:
        ds = Dataset(
            f"{ds.name}|q={args.queries}", ds.train,
            np.ascontiguousarray(ds.queries[: args.queries]),
            np.ascontiguousarray(ds.neighbors[: args.queries]),
            np.ascontiguousarray(ds.distances[: args.queries]),
        )
    return ds


def build_and_measure(ds: Dataset, args: argparse.Namespace, cfg: dict):
    idx, build_s = build_or_load_index(
        ds.train, ds.dim, ds.name, m=args.m, efc=args.efc, seed=0,
        reuse=args.reuse_index, verbose_every=0, **cfg,
    )
    build_dists = idx.counter.count
    return idx, build_s, build_dists


def run_child(args: argparse.Namespace) -> None:
    """子进程入口：只跑一个邻居配置，可选连做 eps 扫描，结果以标记行写到 stdout。

    刻意**不用临时文件**回传结果：这台机器的执行沙箱会拦截删除动作
    （safe-delete 在回收站不可用时会直接抛错），临时文件生命周期很容易踩坑。
    stdout 回传没有这个问题，也少一处状态。
    """
    label, cfg = NEIGHBOR_CONFIGS[args.config_idx]
    ds = load_dataset(args)
    idx, build_s, bd = build_and_measure(ds, args, cfg)
    r = benchmark_search(idx, ds.queries, ds.neighbors, args.k, ef=args.ef)
    st = idx.edge_stats()
    row_a = {
        "配置": label,
        "构图距离计算": float(bd),
        "边数": float(st["edges"]),
        "平均出度": float(st["avg_degree"]),
        "Recall@k": r.recall,
        "QPS": r.qps,
        "距离计算/次查询": r.dists_per_query,
        "构建(s)": build_s if build_s else None,   # None = 命中缓存，不是"零耗时"
    }
    rows_b: list[dict] = []
    if args.with_eps:
        # 单趟采集原始量，再把相对量算出来 —— 避免为了定位基线多跑一遍查询集。
        raw = []
        for eps in args.eps:
            idx.early_stop_eps = float(eps)
            rr = benchmark_search(idx, ds.queries, ds.neighbors, args.k, ef=args.ef)
            raw.append((float(eps), rr))
        idx.early_stop_eps = 0.0                    # 还原，别影响后续复用

        # 基线取 eps 最接近 0 的那一档（通常就是 0），保证相对量有明确参照
        eps0, rr0 = min(raw, key=lambda t: abs(t[0]))
        base_recall, base_cost = rr0.recall, rr0.dists_per_query

        for eps, rr in raw:
            rows_b.append({
                "eps": eps,
                "终止方向": ("收紧（省算力）" if eps < 0 else
                             "基线" if eps == eps0 else "放宽（搜更深）"),
                "Recall@k": rr.recall,
                "召回相对基线差": rr.recall - base_recall,
                "距离计算/次查询": rr.dists_per_query,
                "算力相对节省": (1.0 - rr.dists_per_query / base_cost) if base_cost else 0.0,
                "QPS": rr.qps,
            })
    payload = json.dumps({"row_a": row_a, "rows_b": rows_b}, ensure_ascii=False)
    print(f"{RESULT_MARK}{payload}", flush=True)


def _drain(proc) -> tuple[str, int]:
    out, _ = proc.communicate()
    return out or "", proc.returncode


def spawn_children(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """父进程：把 5 个配置分发给子进程并行跑，用线程并发 drain stdout 回收结果。

    为什么用线程 drain：子进程输出走管道，若父进程逐个 wait 而不读，管道缓冲写满后
    子进程会阻塞（死锁）。communicate() 会释放 GIL，所以多线程能真正并发。
    """
    script = os.path.abspath(__file__)
    base = [sys.executable, script]
    if args.synthetic:
        base.append("--synthetic")
    base += ["--dataset", args.dataset, "--n", str(args.n), "--queries", str(args.queries),
             "--k", str(args.k), "--ef", str(args.ef), "--m", str(args.m), "--efc", str(args.efc)]
    if not args.reuse_index:
        base.append("--no-reuse-index")

    procs = []
    for i, (label, _cfg) in enumerate(NEIGHBOR_CONFIGS):
        cmd = base + ["--config-idx", str(i), "--eps", *[str(e) for e in args.eps]]
        if i == DEFAULT_CONFIG_IDX:
            cmd.append("--with-eps")
        print(f"  启动配置 {i}: {label}", flush=True)
        procs.append((i, label, subprocess.Popen(
            cmd, cwd=os.path.dirname(script), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )))

    payloads: dict[int, dict] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=len(procs)) as ex:
        futures = {ex.submit(_drain, p): (i, label) for i, label, p in procs}
        for fut in as_completed(futures):
            i, label = futures[fut]
            out, rc = fut.result()
            # 把子进程的普通输出也转出来，失败时才看得见原因
            for line in out.splitlines():
                if not line.startswith(RESULT_MARK):
                    print(f"    [{label}] {line}")
            marked = [ln for ln in out.splitlines() if ln.startswith(RESULT_MARK)]
            if rc != 0 or not marked:
                failed.append(f"{label}(rc={rc})")
                continue
            payloads[i] = json.loads(marked[-1][len(RESULT_MARK):])

    if failed:
        # 不允许静默丢配置 —— 消融表少一行会让结论站不住
        raise RuntimeError(f"以下配置未产出结果：{failed}")

    rows_a = [payloads[i]["row_a"] for i in sorted(payloads)]
    rows_b: list[dict] = []
    for i in sorted(payloads):
        if payloads[i]["rows_b"]:
            rows_b = payloads[i]["rows_b"]
    return rows_a, rows_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=80)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=100)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[-0.75, -0.5, -0.25, 0.0, 0.25, 0.5],
                    help="双向扫描：负值收紧阈值（省算力），正值放宽阈值（搜更深），"
                         "取值必须 > -1")
    ap.add_argument("--reuse-index", action="store_true", default=True)
    ap.add_argument("--no-reuse-index", dest="reuse_index", action="store_false")
    ap.add_argument("--config-idx", type=int, default=-1, help="内部使用：子进程只跑某个配置")
    ap.add_argument("--with-eps", action="store_true", help="内部使用：子进程顺带做 eps 扫描")
    args = ap.parse_args()

    if args.config_idx >= 0:                       # 子进程分支
        run_child(args)
        return

    ds = load_dataset(args)
    banner(f"数据：{ds.describe()}｜m={args.m} ef_construction={args.efc} ef={args.ef}")

    banner("消融 A/B：5 个邻居配置并行构图")
    t0 = time.perf_counter()
    rows_a, rows_b = spawn_children(args)
    print(f"\n5 个配置全部完成，墙钟 {time.perf_counter() - t0:.1f}s")

    cols_a = [
        ("配置", "配置"), ("构图距离计算", "构图距离计算"), ("边数", "边数"),
        ("平均出度", "平均出度"), ("Recall@k", "Recall@k"), ("QPS", "QPS"),
        ("距离计算/次查询", "距离计算/次查询"), ("构建(s)", "构建(s)"),
    ]
    table_a = format_table(rows_a, cols_a, digits=3)
    print("\n" + table_a)
    save_rows("exp2a_neighbor_selection", rows_a)

    cols_b = [
        ("eps", "eps"), ("终止方向", "终止方向"), ("Recall@k", "Recall@k"),
        ("召回相对基线差", "召回相对基线差"), ("距离计算/次查询", "距离计算/次查询"),
        ("算力相对节省", "算力相对节省"), ("QPS", "QPS"),
    ]
    table_b = format_table(rows_b, cols_b, digits=4)
    print("\n" + table_b)
    save_rows("exp2b_early_stop_eps", rows_b)

    with open(os.path.join(RESULTS, "exp2_ablation.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# 消融实验\n\n数据：{ds.describe()}｜m={args.m} "
                 f"ef_construction={args.efc} ef={args.ef}\n\n"
                 f"## A 邻居选择策略\n\n{table_a}\n\n"
                 f"## B (1+eps) 终止阈值（双向扫描）\n\n{table_b}\n\n"
                 "> 判据为 `候选距离 > 第 ef 名距离 × (1+eps)`，作用在**平方距离**上。\n"
                 "> **eps 越大不是越早停**：正 eps 放宽阈值（搜更深、算力升），"
                 "负 eps 收紧阈值（更早停、算力降）。\n"
                 "> 本组只改查询期 eps，6 个档位共用同一张图，差异全部来自搜索期终止策略。\n")
    print("\n已保存 results/exp2a_neighbor_selection.csv / exp2b_early_stop_eps.csv / exp2_ablation.md")


if __name__ == "__main__":
    main()
