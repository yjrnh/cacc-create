"""实验五：BCAS 尾部失败的归因诊断（报告 4.6 节负结果归因）。

回答一个问题：**为什么 BCAS 在 n=5 万 上没能改善 P1？**

表 4-2 已经给出了「P1 恒为 0」这个负结果，但只报负结果不报根因是
不合格的 —— 评审会问"是方法错了还是实验设计错了"。本脚本把根因拆成
三个互不重叠、各自可独立否证的问题：

D1 池化臂的 cap 记账复核
    表 4-2 的每一个池化数字都依赖「批次共享预算池」这套记账，口径必须先钉死。
    exp4 打印的「触顶查询 = sum(caps >= cap_max)」在 n=5 万 上给出的数与
    caps 的实际最大值矛盾，因此这里不看结论、只看原始分布：caps 的最小值 /
    分位数 / 最大值，以及 sum(caps >= cap_max)、sum(caps >= B) 两个计数。
    口径确认之前，表 4-2 里任何"再分配发生了"的措辞都不能写。

D2 失败查询的算力画像（检验「早停反向惩罚」假设）
    边际收益早停判据 g_r = (d_k^(r-1) - d_k^(r)) / d_k^(r-1) < τ 只看 top-k
    的距离是否还在降，不看"这个查询是否已经找对了人"。
    如果假设成立，则：
      - 找不到近邻的查询，d_k 一直不降 → g_r 小 → 被早停掐掉；
      - 早早找对的查询，前几轮 d_k 快速下降 → g_r 大 → 被允许继续花钱。
    即"困难查询拿到的算力反而更少"。这是可以用相关系数直接检验的。

D3 失败的两种解释，分开检验（关键分支）
    解释 A（可达性）：这些查询的近邻在图里根本走不到 → 无论给多少算力都不恢复。
    解释 B（近邻对比度）：近邻之间的距离过于接近（近值并列），图返回的点距离几乎
      相同但编号不对 → 加大 ef 会部分恢复；即使不恢复，返回距离与真值也几乎相等。
    这两种解释对项目含义完全不同：A 说明是图结构问题（该项目无法解决，须如实承认），
    B 说明是**近邻鉴别问题**，而它恰恰可以用"把算力花在鉴别上"来缓解 ——
    即 BCAS 的增量方向依然成立，只是当前早停判据选错了停止时机。
    检验方式：
      - 失败查询在 ef ∈ {160, 320, 640, 1024, 2048} 下的恢复数（判断 A）；
      - d_returned_k / d_true_k 的比值（判断 B 的指纹；比值 ≈ 1 即近值并列）；
      - 与满分查询做对照组，比值差异即鉴别难度差异。

用法：
    python experiments/exp5_failure_diag.py --synthetic --n 50000 --queries 500 \
        --k 10 --m 16 --efc 100 --reuse-index \
        --gain-tau 0.0 0.01 0.2 --pool-cap-factor 8 --tag exp5_failure_diag
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from _common import RESULTS, banner, build_or_load_index, save_rows
from src.adaptive import AdaptiveSearcher, PooledBudgetSearcher, pooled_batch_eval
from src.data import Dataset, load_hdf5, make_synthetic
from src.metrics import format_table


# ---------------- 小工具 ----------------

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """秩相关系数。手写避免引入 scipy 依赖（本项目只用 numpy）。"""
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den else float("nan")


def kth_returned_distance(dst: np.ndarray, k: int) -> float:
    """返回结果里第 k 名的平方距离；不足 k 条返回 nan（不足 k 说明图只返回了更少）。"""
    return float(dst[k - 1]) if dst.size >= k else float("nan")


def run_fixed_ef_subset(idx, ds, subset, ef: int, k: int):
    """对指定查询子集做固定 ef 搜索。返回 (recalls, dists, dreturned)。"""
    n = len(subset)
    recalls = np.zeros(n)
    dists = np.zeros(n, dtype=np.int64)
    dret = np.full(n, np.nan)
    for j, i in enumerate(subset):
        base = idx.counter.count
        ids, dst = idx.search(ds.queries[i], k, ef=ef)
        dists[j] = idx.counter.count - base
        hit = len(set(int(x) for x in ids[:k]) & set(int(x) for x in ds.neighbors[i, :k]))
        recalls[j] = hit / float(k)
        dret[j] = kth_returned_distance(dst, k)
    return recalls, dists, dret


# ---------------- 数据与索引 ----------------

def load_dataset(args: argparse.Namespace) -> Dataset:
    banner("载入数据")
    if args.synthetic:
        ds = make_synthetic(n=args.n, dim=64, nq=args.queries, k=args.k)
    else:
        ds = load_hdf5(args.dataset)
        if args.n < ds.n:
            banner(f"底库子采样至 {args.n}，重算 ground truth（这一步必须做）")
            ds = ds.with_train_subset(args.n, recompute_truth=True)
        if args.queries < ds.nq:
            ds = Dataset(
                f"{ds.name}|q={args.queries}", ds.train,
                np.ascontiguousarray(ds.queries[: args.queries]),
                np.ascontiguousarray(ds.neighbors[: args.queries]),
                np.ascontiguousarray(ds.distances[: args.queries]),
            )
    print(ds.describe())
    return ds


def build_or_load(ds: Dataset, args: argparse.Namespace):
    banner("准备索引")
    idx, bt = build_or_load_index(
        ds.train, ds.dim, ds.name, m=args.m, efc=args.efc, seed=0,
        reuse=args.reuse_index, verbose_every=max(1, ds.n // 10),
    )
    bt_txt = f"{bt:.1f}s" if bt is not None else "未知（缓存无元数据）"
    print(f"构建耗时 {bt_txt} | 构图距离计算 {idx.counter.count:,}")
    print(idx.memory_report())
    return idx


# ---------------- 各臂的逐查询采样 ----------------

def run_fixed_ef(idx, ds, ef: int, k: int):
    """固定 ef 逐查询采样。返回 (recalls, dists, dreturned)。"""
    nq = ds.queries.shape[0]
    recalls = np.zeros(nq)
    dists = np.zeros(nq, dtype=np.int64)
    dret = np.full(nq, np.nan)
    for i in range(nq):
        base = idx.counter.count
        ids, dst = idx.search(ds.queries[i], k, ef=ef)
        dists[i] = idx.counter.count - base
        hit = len(set(int(x) for x in ids[:k]) & set(int(x) for x in ds.neighbors[i, :k]))
        recalls[i] = hit / float(k)
        dret[i] = kth_returned_distance(dst, k)
    return recalls, dists, dret


def run_perquery(idx, ds, k: int, budget: int, tau: float, ef0: int, ef_max: int, warmup: int = 3):
    """BCAS-复用臂的逐查询采样。warmup 与 exp4 一致，保证与表 4-2 同口径。"""
    s = AdaptiveSearcher(idx, k=k, budget=budget, ef0=ef0, ef_max=ef_max,
                         gain_tau=tau, resume=True)
    nq = ds.queries.shape[0]
    for i in range(min(warmup, nq)):
        s.search(ds.queries[i])
    recalls = np.zeros(nq)
    dists = np.zeros(nq, dtype=np.int64)
    rounds = np.zeros(nq, dtype=np.int64)
    dret = np.full(nq, np.nan)
    for i in range(nq):
        ids, dst = s.search(ds.queries[i])
        dists[i] = s.last_dists
        rounds[i] = s.last_rounds
        dret[i] = kth_returned_distance(dst, k)
        hit = len(set(int(x) for x in ids[:k]) & set(int(x) for x in ds.neighbors[i, :k]))
        recalls[i] = hit / float(k)
    return recalls, dists, rounds, dret


def run_pooled(idx, ds, k: int, budget: int, tau: float, cap_factor: float,
               ef0: int, ef_max: int):
    """池化臂逐查询采样。返回 (recalls, dists, caps, rounds, dret, 池统计)。"""
    ps = PooledBudgetSearcher(idx, k=k, budget=budget, cap_factor=cap_factor,
                              ef0=ef0, ef_max=ef_max, gain_tau=tau)
    nq = ds.queries.shape[0]
    ps.begin_batch(nq)
    recalls = np.zeros(nq)
    dists = np.zeros(nq, dtype=np.int64)
    caps = np.zeros(nq, dtype=np.int64)
    rounds = np.zeros(nq, dtype=np.int64)
    dret = np.full(nq, np.nan)
    for i in range(nq):
        ids, dst = ps.search(ds.queries[i])
        dists[i] = ps.last_dists
        caps[i] = ps.last_cap
        rounds[i] = ps.last_rounds
        dret[i] = kth_returned_distance(dst, k)
        hit = len(set(int(x) for x in ids[:k]) & set(int(x) for x in ds.neighbors[i, :k]))
        recalls[i] = hit / float(k)
    stat = {
        "pool_total": int(ps.pool_total), "pool_remaining": int(ps.pool_remaining),
        "cap_max": int(ps.cap_max), "floor": int(ps.floor),
        "starved": int(ps.starved), "overshoot": int(ps.overshoot),
        "overdraft": int(ps.overdraft), "caps_at_capmax": int((caps >= ps.cap_max).sum()),
        "caps_ge_B": int((caps >= budget).sum()),
    }
    return recalls, dists, caps, rounds, dret, stat


# ---------------- 主流程 ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=100)
    ap.add_argument("--ef-grid", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    ap.add_argument("--diag-ef", type=int, nargs="+", default=[160, 320, 640, 1024, 2048],
                    help="D3 可达性上限扫描：把失败查询交给这些固定 ef，看能否被救回")
    ap.add_argument("--budgets", type=int, nargs="+", default=[425, 643, 972, 1470])
    ap.add_argument("--gain-tau", type=float, nargs="+", default=[0.0, 0.01, 0.2])
    ap.add_argument("--ef0", type=int, default=16)
    ap.add_argument("--ef-max", type=int, default=1024)
    ap.add_argument("--pool-cap-factor", type=float, default=8.0)
    ap.add_argument("--reuse-index", action="store_true", default=True)
    ap.add_argument("--no-reuse-index", dest="reuse_index", action="store_false")
    ap.add_argument("--tag", default="exp5_failure_diag")
    args = ap.parse_args()

    ds = load_dataset(args)
    idx = build_or_load(ds, args)
    k = args.k
    nq = ds.queries.shape[0]
    truth_k = ds.distances[:, k - 1].astype(np.float64)   # 真值第 k 名平方距离

    rows: list[dict] = []

    # ---- 基准：固定 ef 曲线（同时给出各 ef 的零召回集合） ----
    banner("D0 基准：固定 ef 曲线与零召回集合")
    curve = []
    rec_sets: dict[str, np.ndarray] = {}
    for ef in args.ef_grid:
        rec, cst, dret = run_fixed_ef(idx, ds, ef, k)
        zero = rec == 0.0
        rec_sets[f"fixed_ef{ef}"] = rec
        ratio = dret / truth_k
        ok = rec >= 1.0
        curve.append({
            "ef": ef, "cost": float(cst.mean()), "recall": float(rec.mean()),
            "zero_recall": int(zero.sum()),
            "零召回组算力/全体": float(cst[zero].mean() / cst.mean()) if zero.any() else float("nan"),
            "满分组算力/全体": float(cst[ok].mean() / cst.mean()) if ok.any() else float("nan"),
            "r轮次_零召回": float("nan"), "r轮次_满分": float("nan"),
            "比值d_returned/d_true_零召回": float(np.nanmean(ratio[zero])) if zero.any() else float("nan"),
            "比值d_returned/d_true_满分": float(np.nanmean(ratio[ok])) if ok.any() else float("nan"),
            "spearman(算力,召回)": spearman(cst.astype(np.float64), rec),
        })
        print(f"ef={ef:<5d} cost={cst.mean():8.1f} recall={rec.mean():.4f} "
              f"零召回={int(zero.sum()):3d} 零召回组算力/全体="
              f"{cst[zero].mean() / cst.mean() if zero.any() else float('nan'):.3f} "
              f"ρ(算力,召回)={spearman(cst.astype(np.float64), rec):+.3f}")

    print("\n" + format_table(curve, [
        ("ef", "ef"), ("距离/查询", "cost"), ("Recall@k", "recall"), ("零召回查询数", "zero_recall"),
        ("零召回组算力/全体", "零召回组算力/全体"), ("满分组算力/全体", "满分组算力/全体"),
        ("d_ret/d_true(零召回)", "比值d_returned/d_true_零召回"),
        ("d_ret/d_true(满分)", "比值d_returned/d_true_满分"),
        ("ρ(算力,召回)", "spearman(算力,召回)"),
    ], digits=3))

    # ---- D1 + D2：逐 (tau, 预算) 跑两条臂，记录逐查询画像 ----
    banner("D1/D2 预算臂的逐查询画像与池化 cap 记账复核")
    fail_union: set[int] = set()
    fail_inter: set[int] | None = None
    arm_dump: dict[str, dict] = {}

    for tau in args.gain_tau:
        print(f"\n{'#' * 72}\n# gain_tau = {tau}\n{'#' * 72}")
        for B in args.budgets:
            rec_a, dst_a, rnd_a, dret_a = run_perquery(idx, ds, k, B, tau, args.ef0, args.ef_max)
            rec_p, dst_p, cap_p, rnd_p, dret_p, pstat = run_pooled(
                idx, ds, k, B, tau, args.pool_cap_factor, args.ef0, args.ef_max)

            print(f"  [池化 B={B}] cap 分布 min={cap_p.min()} p25={np.percentile(cap_p, 25):.0f} "
                  f"中位={np.median(cap_p):.0f} p75={np.percentile(cap_p, 75):.0f} max={cap_p.max()} "
                  f"| cap_max={pstat['cap_max']} 保底={pstat['floor']} "
                  f"| sum(cap>=cap_max)={pstat['caps_at_capmax']} sum(cap>=B)={pstat['caps_ge_B']} "
                  f"| 池总额={pstat['pool_total']:,} 剩余={pstat['pool_remaining']:,} "
                  f"starved={pstat['starved']} 透支={pstat['overdraft']}")

            for label, rec, dst, rnd, dret in (
                ("BCAS-复用", rec_a, dst_a, rnd_a, dret_a),
                ("BCAS-池化", rec_p, dst_p, rnd_p, dret_p),
            ):
                zero = rec == 0.0
                ok = rec >= 1.0
                fail_union |= set(np.nonzero(zero)[0].tolist())
                fail_inter = set(np.nonzero(zero)[0].tolist()) if fail_inter is None \
                    else (fail_inter & set(np.nonzero(zero)[0].tolist()))
                keys = {
                    "tau": tau, "B": B, "方案": label,
                    "Recall均值": float(rec.mean()), "零召回查询数": int(zero.sum()),
                    "满分查询数": int(ok.sum()),
                    "算力均值": float(dst.mean()),
                    "零召回组算力": float(dst[zero].mean()) if zero.any() else float("nan"),
                    "零召回组算力/全体": float(dst[zero].mean() / dst.mean()) if zero.any() else float("nan"),
                    "满意组算力/全体": float(dst[ok].mean() / dst.mean()) if ok.any() else float("nan"),
                    "零召回组轮次": float(rnd[zero].mean()) if zero.any() else float("nan"),
                    "满分组轮次": float(rnd[ok].mean()) if ok.any() else float("nan"),
                    "全体轮次": float(rnd.mean()),
                    "d_ret/d_true(零召回)": float(np.nanmean((dret / truth_k)[zero])) if zero.any() else float("nan"),
                    "d_ret/d_true(满分)": float(np.nanmean((dret / truth_k)[ok])) if ok.any() else float("nan"),
                    "spearman(算力,召回)": spearman(dst.astype(np.float64), rec),
                }
                rows.append(keys)
                arm_dump[f"tau{tau}_B{B}_{label}"] = {
                    "recalls": rec.tolist(), "dists": dst.tolist(),
                    "rounds": rnd.tolist(), "dret": dret.tolist(),
                }
                print(f"    {label:9s} recall={rec.mean():.4f} 零召回={int(zero.sum()):3d} "
                      f"算力={dst.mean():8.1f} 零召回组算力={keys['零召回组算力']:8.1f}"
                      f"（{keys['零召回组算力/全体']:.3f}×） "
                      f"零召回组轮次={keys['零召回组轮次']:.2f} 全体轮次={rnd.mean():.2f} "
                      f"d_ret/d_true={keys['d_ret/d_true(零召回)']:.3f} "
                      f"ρ={keys['spearman(算力,召回)']:+.3f}")

    fail_union_idx = sorted(fail_union)
    fail_inter_idx = sorted(fail_inter or set())
    print(f"\n零召回的并集 {len(fail_union_idx)} 条，交集 {len(fail_inter_idx)} 条"
          f"（交集=在所有臂、所有预算下都失败的查询）")

    # ---- D3：可达性上限 + 近邻对比度 ----
    banner("D3 失败查询的可达性上限与近邻对比度")
    probe = fail_union_idx if len(fail_union_idx) >= 5 else sorted(
        np.nonzero(rec_sets[f"fixed_ef{args.ef_grid[len(args.ef_grid) // 2]}"] == 0.0)[0].tolist())
    # 对照组：在最高 ef 下满分的查询，取与探针同数量。它是"容易的查询"的基线，
    # 用来回答"失败组的近邻对比度是否真的更差"，而不是只看失败组自己的数。
    control = sorted(np.nonzero(rec_sets[f"fixed_ef{args.ef_grid[-1]}"] >= 1.0)[0].tolist())[:len(probe)]
    print(f"探针集合：并集失败 {len(fail_union_idx)} 条（用 {len(probe)} 条）；"
          f"对照组：ef={args.ef_grid[-1]} 下满分的 {len(control)} 条")

    d3_rows = []
    for ef in args.diag_ef:
        rec, cst, dret = run_fixed_ef_subset(idx, ds, probe, ef, k)
        crec, ccst, cdret = run_fixed_ef_subset(idx, ds, control, ef, k)
        d3_rows.append({
            "ef": ef, "距离/查询": float(cst.mean()),
            "失败组恢复数": int((rec > 0).sum()), "失败组样本": len(probe),
            "失败组恢复率": float((rec > 0).mean()),
            "失败组Recall均值": float(rec.mean()),
            "失败组d_ret/d_true": float(np.nanmean(dret / truth_k[probe])),
            "对照组Recall均值": float(crec.mean()),
            "对照组d_ret/d_true": float(np.nanmean(cdret / truth_k[control])),
        })
        print(f"ef={ef:<5d} 算力={cst.mean():9.1f} 失败组恢复 {int((rec > 0).sum())}/{len(probe)} "
              f"Recall={rec.mean():.4f} d_ret/d_true={np.nanmean(dret / truth_k[probe]):.4f} "
              f"| 对照组 Recall={crec.mean():.4f} d_ret/d_true={np.nanmean(cdret / truth_k[control]):.4f}")

    print("\n" + format_table(d3_rows, [
        ("ef", "ef"), ("距离/查询", "距离/查询"),
        ("失败组恢复数", "失败组恢复数"), ("失败组样本", "失败组样本"),
        ("失败组恢复率", "失败组恢复率"), ("失败组Recall", "失败组Recall均值"),
        ("失败组d_ret/d_true", "失败组d_ret/d_true"),
        ("对照组Recall", "对照组Recall均值"), ("对照组d_ret/d_true", "对照组d_ret/d_true"),
    ], digits=4))

    # 真值本身的对比度：第 k 名与第 1 名的距离之比。比值越接近 1，鉴别越难。
    banner("D3b 真值近邻对比度（决定失败是鉴别问题还是可达性问题）")
    contrast = truth_k / ds.distances[:, 0].astype(np.float64)
    for name, group in (("失败并集", fail_union_idx), ("满分对照", control)):
        if not group:
            continue
        g = np.asarray(group)
        print(f"  {name:6s} n={g.size:3d} 真值 d_k/d_1：均值={contrast[g].mean():.4f} "
              f"中位={np.median(contrast[g]):.4f} max={contrast[g].max():.4f}")

    # ---- 落盘 ----
    out = {"args": vars(args),
           "curve": curve,
           "rows": rows,
           "d3": d3_rows,
           "fail_union": fail_union_idx,
           "fail_inter": fail_inter_idx,
           "control": control,
           "contrast": {"fail": float(contrast[fail_union_idx].mean()) if fail_union_idx else None,
                        "control": float(contrast[control].mean()) if control else None},
           "arm_dump": arm_dump}
    path = save_rows(args.tag, rows) if rows else ""
    json_path = os.path.join(RESULTS, f"{args.tag}_detail.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, default=float)
    if path:
        print(f"\n已保存 {path}")
    print(f"已保存 {json_path}（含逐查询召回/算力/轮次/d_returned，供第三方复核）")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
