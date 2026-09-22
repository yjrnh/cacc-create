"""实验四：等算力预算下固定 ef 与预算自适应搜索的对比（报告 4.3 节核心表）。

    python experiments/exp4_budget.py --synthetic --n 20000 --queries 300 --k 10 \
        --budgets 300 600 1200 2400 --ef-grid 10 20 40 80 160 320

比较逻辑（关键，决定这张表能不能站住）：
  1. 先扫一遍固定 ef，得到 (平均距离计算次数, 平均召回率, P1, P5, 标准差) 的实测点集；
  2. 对每个预算 B，用 BCAS-复用 的**实测平均算力**去曲线上做线性插值 →
     "花掉同样多算力的固定 ef 能拿多少平均召回"；
  3. 同时在曲线上取**最近算力点**的完整分布统计（P1 / P5 / 标准差）作为固定 ef 的尾部参考。
     均值用插值（更准），尾部用最近点（插值没有分位数含义）—— 两者都标注清楚。

表里每一行都自洽可查：BCAS 的实测值、固定 ef 的实测值、以及两者之间的插值桥。

一个重要且不打算隐藏的发现：**「每轮从零重算」的方案在结构上无法守住硬预算**。
因为单次固定 ef 搜索的开销事前不可知，只能在事后发现超支。因此表里
BCAS-重算 一行的"超预算率"非零，这不是实现缺陷，而是它被淘汰的理由。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from _common import RESULTS, banner, build_or_load_index, save_rows
from src.adaptive import (
    AdaptiveSearcher,
    PooledBudgetSearcher,
    interpolate_recall_at_cost,
    perquery_eval,
    pooled_batch_eval,
)
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
                f"{ds.name}|q={args.queries}", ds.train,
                np.ascontiguousarray(ds.queries[: args.queries]),
                np.ascontiguousarray(ds.neighbors[: args.queries]),
                np.ascontiguousarray(ds.distances[: args.queries]),
            )
    print(ds.describe())
    return ds


def build_or_load(ds: Dataset, args: argparse.Namespace):
    """构图很慢，实验之间必须复用（缓存键含决定图结构的全部参数，见 _common.index_key）。"""
    banner("准备索引")
    idx, bt = build_or_load_index(
        ds.train, ds.dim, ds.name, m=args.m, efc=args.efc, seed=0,
        reuse=args.reuse_index, verbose_every=max(1, ds.n // 10),
    )
    bt_txt = f"{bt:.1f}s" if bt is not None else "未知（缓存无元数据）"
    print(f"构建耗时 {bt_txt} | 构图距离计算 {idx.counter.count:,}")
    print(idx.memory_report())
    return idx, bt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=100)
    ap.add_argument("--ef-grid", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    ap.add_argument("--budgets", type=int, nargs="+", default=[300, 600, 1200, 2400])
    ap.add_argument("--auto-budgets", type=int, default=0,
                    help=">0 时忽略 --budgets，改在固定 ef 实测曲线的算力区间内按对数等距取这么多点。"
                         "这样保证每个预算都能在曲线上插值，而不是外推。")
    ap.add_argument("--gain-tau", type=float, nargs="+", default=[0.01],
                    help="边际收益早停阈值；可给多个值，固定 ef 曲线只扫一次，"
                         "同一张表即可读出 tau 敏感性")
    ap.add_argument("--pool", action="store_true",
                    help="额外跑一条「批次共享预算池」臂（BCAS-P）：总额度 nq×B，"
                         "每查询保底 B、上限 cap_factor×B。这是唯一能让"
                         "「预算再分配」在结构上成立的记账方式。")
    ap.add_argument("--pool-cap-factor", type=float, default=8.0,
                    help="池化臂的单查询算力上限倍数（cap_max = factor × B）")
    ap.add_argument("--ef0", type=int, default=16)
    ap.add_argument("--ef-max", type=int, default=1024)
    ap.add_argument("--reuse-index", action="store_true", default=True)
    ap.add_argument("--no-reuse-index", dest="reuse_index", action="store_false")
    ap.add_argument("--keep-perquery", action="store_true", default=True,
                    help="把逐查询召回/算力写进 JSON，第三方可据此重画尾部曲线")
    ap.add_argument("--tag", default="exp4_budget")
    args = ap.parse_args()

    ds = load_dataset(args)
    idx, build_s = build_or_load(ds, args)

    # ---- 第一步：固定 ef 支线（等算力插值的基准曲线 + 尾部参考点） ----
    banner("固定 ef 扫描：等算力基准曲线")
    curve: list[dict] = []
    for ef in args.ef_grid:
        st = perquery_eval(lambda q, ef=ef: idx.search(q, args.k, ef=ef),
                           ds.queries, ds.neighbors, args.k, idx.counter)
        curve.append({
            "ef": ef,
            "cost": st["dists_mean"],
            "recall_mean": st["recall_mean"],
            "p1": st["recall_p1"],
            "p5": st["recall_p5"],
            "std": st["recall_std"],
            "cost_p99": st["dists_p99"],
            "cost_max": st["dists_max"],
            "zero_recall": int((np.asarray(st["recalls"]) == 0.0).sum()),
        })
        print(f"ef={ef:<4d} cost={st['dists_mean']:9.1f} recall={st['recall_mean']:.4f} "
              f"p1={st['recall_p1']:.4f} p5={st['recall_p5']:.4f} std={st['recall_std']:.4f}")

    efs = [c["ef"] for c in curve]
    costs = [c["cost"] for c in curve]
    recalls = [c["recall_mean"] for c in curve]
    curve_cols = [("ef", "ef"), ("Recall@k 均值", "recall_mean"), ("Recall P1", "p1"),
                  ("Recall P5", "p5"), ("Recall 标准差", "std"),
                  ("距离/查询", "cost"), ("距离 P99", "cost_p99")]
    curve_table = format_table(curve, curve_cols, digits=4)
    print("\n" + curve_table)

    # ---- 第二步：逐 (tau, 预算) 对比 ----
    # tau 敏感性分析如果各跑一遍实验，就要把上面那条固定 ef 曲线重扫 N 遍 ——
    # 那是纯粹浪费。曲线只依赖图与查询集，与 tau 无关，因此在这里循环 tau、
    # 只扫一次曲线。表里用 gain_tau 一列区分，同一张表即可读出敏感性。
    banner("预算约束对比")

    # 预算必须落在固定 ef 实测曲线的**算力区间内**，否则"等算力插值召回"是外推值，
    # 拿它当基准去比 BCAS 是无效比较（比的是"曲线外没有的数据"而不是分配策略）。
    budgets = list(args.budgets)
    if args.auto_budgets:
        lo, hi = min(costs), max(costs)
        # 往区间内缩 15%：BCAS 的实际算力是"预算利用率 ~0.95"的均值，
        # 会略低于 B，缩得太少会让最低那档预算落到曲线下界之外（变成外推行）。
        grid = np.exp(np.linspace(np.log(lo * 1.15), np.log(hi * 0.90), args.auto_budgets))
        budgets = sorted({int(round(x)) for x in grid})
        print(f"自动取预算：{budgets}（固定 ef 曲线算力区间 {lo:.0f} ~ {hi:.0f}）")
    print("每行都会判定「对齐有效性」：插值落在曲线区间内才记「有效」，"
          "「外推」行的召回差不作为结论依据。")

    main_rows: list[dict] = []
    detail: dict = {"curve": curve, "budgets": {}}

    for tau in args.gain_tau:
        print(f"\n{'#' * 72}\n# gain_tau = {tau}\n{'#' * 72}")
        for B in budgets:
            measured: dict[str, dict] = {}
            for label, resume in (("BCAS-复用", True), ("BCAS-重算", False)):
                searcher = AdaptiveSearcher(
                    idx, k=args.k, budget=B, ef0=args.ef0, ef_max=args.ef_max,
                    gain_tau=tau, resume=resume,
                )
                rounds: list[int] = []
                raw = searcher.search

                def fn(q, _s=searcher, _raw=raw):
                    ids, dst = _raw(q)
                    rounds.append(_s.last_rounds)
                    return ids, dst

                st = perquery_eval(fn, ds.queries, ds.neighbors, args.k, idx.counter)
                over = float(np.mean(np.asarray(st["dists"]) > B))
                # 硬预算守卫只对「复用状态」这条路成立；从零重算的方案结构上无法保证
                # （单次固定 ef 搜索的开销事前不可知），这一点必须如实报出而不是抹平。
                if resume:
                    assert over == 0.0, f"硬预算被突破：{label} 预算={B} 超预算比例={over}"
                measured[label] = {
                    "st": st, "rounds": rounds, "over": over,
                    "rounds_mean": float(np.mean(rounds)) if rounds else 0.0,
                }

            # 池化臂：预算不是逐查询而是批次共享。这条臂的唯一目的，
            # 是检验「早停省下的额度能否真的被困难查询用掉」——
            # 逐查询记账下这件事在结构上不可能发生（省下的额度没有接收方）。
            if args.pool:
                psearch = PooledBudgetSearcher(
                    idx, k=args.k, budget=B, cap_factor=args.pool_cap_factor,
                    ef0=args.ef0, ef_max=args.ef_max, gain_tau=tau,
                )
                pst = pooled_batch_eval(psearch, ds.queries, ds.neighbors, args.k)
                measured["BCAS-池化"] = {
                    "st": pst, "rounds": [], "over": pst["overdraft"] / float(pst["pool_total"]),
                    "rounds_mean": pst["rounds_mean"],
                }
                print(f"  [池化] 池总额={pst['pool_total']:,} 剩余={pst['pool_remaining']:,} "
                      f"单查询上限={psearch.cap_max} 触顶查询={int((pst['caps'] >= psearch.cap_max).sum())} "
                      f"保底不足={pst['starved']} 超上限(下降段)={pst['overshoot']}(max +{pst['overshoot_max']}) "
                      f"池透支={pst['overdraft']}")

            ref_cost = measured["BCAS-复用"]["st"]["dists_mean"]
            interp, note = interpolate_recall_at_cost(efs, costs, recalls, ref_cost)
            nearest = min(curve, key=lambda r: abs(r["cost"] - ref_cost))
            # 插值说明里出现"低于/高于区间"就是外推：此时 interp 只是端点值，
            # 与 ref_cost 根本不在同一算力上，不能拿来算"召回差"。
            valid = ("低于" not in note) and ("高于" not in note)
            print(f"\ntau={tau}  B={B}  对齐基准：BCAS-复用 实测算力 {ref_cost:.1f}"
                  f"（固定 ef 插值召回 {interp:.4f} [{note}]"
                  f"{'' if valid else ' ← 外推，本行召回差不作依据'}；"
                  f"最近算力点 ef={nearest['ef']} 实测 {nearest['cost']:.1f}）")

            arm_labels = ["BCAS-复用", "BCAS-重算"] + (["BCAS-池化"] if args.pool else [])
            for label in arm_labels:
                m = measured[label]
                st = m["st"]
                dispersion = (st["dists_max"] / st["dists_mean"]) if st["dists_mean"] else float("nan")
                main_rows.append({
                    "gain_tau": tau, "预算B": B, "方案": label, "对齐ef": "-",
                    "对齐有效性": "有效" if valid else "外推",
                    "Recall@k 均值": st["recall_mean"],
                    "等算力插值召回": interp,
                    "召回差(BCAS-插值)": st["recall_mean"] - interp if valid else "-",
                    "Recall P1": st["recall_p1"],
                    "P1差(BCAS-最近算力点)": st["recall_p1"] - nearest["p1"],
                    "Recall P5": st["recall_p5"],
                    "Recall 标准差": st["recall_std"],
                    "零召回查询数": int((np.asarray(st["recalls"]) == 0.0).sum()),
                    "距离/查询(实测)": st["dists_mean"],
                    "算力离散度(max/mean)": dispersion,
                    "预算利用率": st["dists_mean"] / float(B),
                    "超预算率": m["over"],
                    "平均轮次": m["rounds_mean"],
                })
                print(f"  {label}  recall={st['recall_mean']:.4f} p1={st['recall_p1']:.4f} "
                      f"p5={st['recall_p5']:.4f} std={st['recall_std']:.4f} "
                      f"cost={st['dists_mean']:8.1f} 离散度={dispersion:5.2f} "
                      f"利用率={st['dists_mean']/B:.3f} "
                      f"超预算率={m['over']:.3f} 轮次={m['rounds_mean']:.2f}")

            main_rows.append({
                "gain_tau": tau, "预算B": B,
                "方案": f"固定ef(最近算力点 ef={nearest['ef']})", "对齐ef": nearest["ef"],
                "对齐有效性": "有效" if valid else "外推",
                "Recall@k 均值": nearest["recall_mean"],
                "等算力插值召回": interp,
                "召回差(BCAS-插值)": 0.0 if valid else "-",
                "Recall P1": nearest["p1"],
                "P1差(BCAS-最近算力点)": 0.0,
                "Recall P5": nearest["p5"],
                "Recall 标准差": nearest["std"],
                "零召回查询数": nearest["zero_recall"],
                "距离/查询(实测)": nearest["cost"],
                "算力离散度(max/mean)": nearest["cost_max"] / nearest["cost"],
                "预算利用率": nearest["cost"] / float(B),
                "超预算率": 0.0,
                "平均轮次": 1.0,
            })
            print(f"  固定ef(ef={nearest['ef']})  recall={nearest['recall_mean']:.4f} "
                  f"p1={nearest['p1']:.4f} p5={nearest['p5']:.4f} "
                  f"std={nearest['std']:.4f} cost={nearest['cost']:.1f}")

            entry = {
                "ref_cost": ref_cost, "interp_recall": interp, "interp_note": note,
                "nearest_ef": nearest["ef"], "nearest_ef_cost": nearest["cost"],
                "valid": bool(valid),
            }
            for label in arm_labels:
                m = measured[label]
                entry[label] = {k: v for k, v in m["st"].items() if k not in ("recalls", "dists", "caps")}
                entry[label]["overshoot_rate"] = m["over"]
                entry[label]["rounds_mean"] = m["rounds_mean"]
                if args.keep_perquery:
                    entry[label]["recalls_perquery"] = np.round(m["st"]["recalls"], 4).tolist()
                    entry[label]["dists_perquery"] = m["st"]["dists"].tolist()
            detail["budgets"][f"tau{tau}_B{B}"] = entry

    cols = [("预算B", "预算B"), ("gain_tau", "gain_tau"), ("方案", "方案"), ("对齐ef", "对齐ef"),
            ("对齐有效性", "对齐有效性"),
            ("Recall@k 均值", "Recall@k 均值"), ("等算力插值召回", "等算力插值召回"),
            ("召回差(BCAS-插值)", "召回差(BCAS-插值)"),
            ("Recall P1", "Recall P1"), ("P1差(BCAS-最近算力点)", "P1差(BCAS-最近算力点)"),
            ("Recall P5", "Recall P5"),
            ("Recall 标准差", "Recall 标准差"),
            ("零召回查询数", "零召回查询数"),
            ("距离/查询(实测)", "距离/查询(实测)"),
            ("算力离散度(max/mean)", "算力离散度(max/mean)"),
            ("预算利用率", "预算利用率"), ("超预算率", "超预算率"),
            ("平均轮次", "平均轮次")]
    table = format_table(main_rows, cols, digits=4)
    print("\n" + table)

    path = save_rows(args.tag, main_rows)
    pool_txt = (f"池化臂：cap_factor={args.pool_cap_factor}（单查询上限 = {args.pool_cap_factor}×B，"
                f"批次总额 = 查询数×B，每查询保底 B）\n" if args.pool else "池化臂：未启用\n")
    with open(path.replace(".csv", ".md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {args.tag}\n\n数据：{ds.describe()}\n\n"
                 f"构图参数：m={args.m} ef_construction={args.efc} seed=0\n"
                 f"BCAS 参数：ef0={args.ef0} ef_max={args.ef_max} growth=2 "
                 f"gain_tau={args.gain_tau}\n"
                 f"{pool_txt}"
                 f"预算 B={budgets}（自动取点：{'是' if args.auto_budgets else '否'}）"
                 f"　固定 ef 网格={args.ef_grid}\n"
                 f"查询数={ds.nq} k={args.k}\n\n"
                 f"## 表 4-2 等算力预算对比\n\n{table}\n\n"
                 f"## 表 4-1b 固定 ef 基准曲线\n\n{curve_table}\n\n"
                 f"## 对齐说明\n\n"
                 f"固定 ef 无法直接对齐预算，因此对每个预算用 BCAS-复用的**实测算力**"
                 f"在基准曲线上做线性插值，得到「等算力下固定 ef 的平均召回」（列"
                 f"「等算力插值召回」）。尾部统计（P1/P5/标准差/零召回查询数）取自曲线上"
                 f"**最近算力点**，因为插值没有分位数含义。\n\n"
                 f"**「对齐有效性」列必须看**：若 BCAS 的实测算力落在固定 ef 基准曲线的"
                 f"算力区间之外，插值只能取端点值，此时两边根本不在同一算力上，"
                 f"该行的「召回差」记为 `-` 且**不作为结论依据**。预算取点因此默认"
                 f"用 `--auto-budgets` 在曲线区间内取，避免产生无效对比行。\n\n"
                 f"## 三条臂的记账口径差异（决定「超预算率」怎么读）\n\n"
                 f"| 方案 | 预算单位 | 硬性质 | 「超预算率」含义 |\n"
                 f"|---|---|---|---|\n"
                 f"| 固定 ef | 无预算概念 | 无上界（算力由 ef 隐含决定，事前不可知） | 恒 0（无预算可超） |\n"
                 f"| BCAS-复用 | **逐查询**，每查询上限 B | `cost(q) ≤ B` 恒成立 | `cost > B` 的查询占比 |\n"
                 f"| BCAS-重算 | 逐查询，但每轮从零重算 | **无**（单次固定 ef 开销事前不可知） | `cost > B` 的查询占比，预期接近 1 |\n"
                 f"| BCAS-池化 | **批次共享池**，总额 nq×B；每查询保底 B、上限 `cap_factor×B` | `Σ cost ≤ nq×B` 且 `cost(q) ≤ cap_max` | `池透支额 / 池总额`（恒 0 才是对的） |\n\n"
                 f"**「算力离散度(max/mean)」是再分配的直接证据**：固定 ef 与逐查询 BCAS 都接近 1"
                 f"（每查询花得几乎一样多），只有池化臂会显著大于 1。"
                 f"若池化臂的离散度仍接近 1，说明早停并没有真的把额度让出去，"
                 f"「预算再分配」这一主张即不成立。\n\n"
                 f"**池化臂不报 QPS**：共享预算池不能做预热查询（预热会消耗池内额度、"
                 f"扰动分配本身），因此其冷启动开销与其它臂不可比。距离计算次数与缓存状态"
                 f"无关，仍可严格比较。\n")
    print(f"\n已保存 {path}")

    with open(path.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "detail": detail, "rows": main_rows},
                  fh, ensure_ascii=False, indent=2, default=float)
    print(f"已保存 {path.replace('.csv', '.json')}（含逐查询召回/算力，供第三方重画尾部曲线）")


if __name__ == "__main__":
    main()
