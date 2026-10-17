"""从 results/ 提取全部对外引用的数字，生成唯一权威数字源。

产出：
  results/authoritative_numbers.json  —— 机器可读（供 PPT / 文档脚本消费）
  results/authoritative_numbers.md    —— 人可读（每个数字标注出处文件 + 列名）

原则：本文件不硬编码任何实验值，全部从 CSV / JSON 读取。
文档与 PPT 中出现的数字必须能在此文件中找到对应键。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

R = Path(__file__).parent / "results"


def read_csv(name: str) -> list[dict]:
    p = R / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build() -> dict:
    out: dict = {}

    # ---------- 实验配置（全部正式表统一口径） ----------
    out["config"] = {
        "dataset": "synthetic(32 clusters, dim=64)",
        "n": 50000,
        "nq": 500,
        "k": 10,
        "m": 16,
        "efc": 100,
        "seed": 0,
    }

    # ---------- exp1：Recall@k – 距离计算前沿 ----------
    e1 = read_csv("exp1_n50k.csv")
    if e1:
        rows = []
        for r in e1:
            rows.append({
                "method": r["方法"],
                "ef": r["ef"],
                "recall": f(r["Recall@k"]),
                "qps": f(r["QPS"]),
                "latency_ms": f(r["延迟(ms)"]),
                "cost_per_query": f(r["距离计算/次查询"]),
                "build_s": f(r["构建(s)"]),
            })
        out["exp1"] = {"rows": rows}
        ours = [x for x in rows if x["method"].startswith("HNSW(ours)")]
        if ours:
            out["exp1"]["build_s"] = ours[0]["build_s"]
            by_ef = {str(x["ef"]): x for x in ours}
            out["exp1"]["ours_by_ef"] = by_ef
        bf = [x for x in rows if x["method"] == "BruteForce"]
        if bf:
            out["exp1"]["bruteforce"] = bf[0]

    # ---------- exp4：等算力预算对比（主干 τ=0.01） ----------
    e4 = read_csv("exp4_n50k_pool.csv")
    if e4:
        arms = []
        for r in e4:
            arms.append({
                "tau": f(r["gain_tau"]),
                "B": int(float(r["预算B"])),
                "arm": r["方案"],
                "align_ef": r["对齐ef"],
                "valid": r["对齐有效性"],
                "recall": f(r["Recall@k 均值"]),
                "interp": f(r["等算力插值召回"]),
                "diff": f(r["召回差(BCAS-插值)"]),
                "p5": f(r["Recall P5"]),
                "zero_recall": f(r["零召回查询数"]),
                "cost": f(r["距离/查询(实测)"]),
                "dispersion": f(r["算力离散度(max/mean)"]),
                "util": f(r["预算利用率"]),
                "over_budget": f(r["超预算率"]),
            })
        out["exp4"] = {"rows": arms}

        main = [x for x in arms if x["tau"] == 0.01]
        out["exp4"]["tau001"] = main
        out["exp4"]["tau0"] = [x for x in arms if x["tau"] == 0.0]
        out["exp4"]["tau02"] = [x for x in arms if x["tau"] == 0.2]

        def pick(rows, arm, B):
            for x in rows:
                if x["arm"] == arm and x["B"] == B:
                    return x
            return None

        pool = {B: pick(main, "BCAS-池化", B) for B in (425, 643, 972, 1470)}
        reuse = {B: pick(main, "BCAS-复用", B) for B in (425, 643, 972, 1470)}
        recompute = {B: pick(main, "BCAS-重算", B) for B in (425, 643, 972, 1470)}
        fixed = {B: next((x for x in main if x["arm"].startswith("固定ef") and x["B"] == B), None)
                 for B in (425, 643, 972, 1470)}
        out["exp4"]["pool_by_B"] = {str(k): v for k, v in pool.items()}
        out["exp4"]["reuse_by_B"] = {str(k): v for k, v in reuse.items()}
        out["exp4"]["recompute_by_B"] = {str(k): v for k, v in recompute.items()}
        out["exp4"]["fixed_by_B"] = {str(k): v for k, v in fixed.items()}

        # 超预算率：哪些配置恒 0，哪些非 0
        zero_rows = [x for x in arms if x["over_budget"] == 0.0]
        nonzero_rows = [x for x in arms if x["over_budget"] and x["over_budget"] > 0]
        out["exp4"]["over_budget_zero_count"] = len(zero_rows)
        out["exp4"]["over_budget_nonzero_count"] = len(nonzero_rows)
        out["exp4"]["over_budget_nonzero_arms"] = sorted({x["arm"] for x in nonzero_rows})
        out["exp4"]["over_budget_nonzero_tau"] = sorted({x["tau"] for x in nonzero_rows})
        out["exp4"]["total_rows"] = len(arms)

        # 离散度区间
        def rng(rows):
            vals = [x["dispersion"] for x in rows if x["dispersion"] is not None]
            return [round(min(vals), 2), round(max(vals), 2)] if vals else None
        out["exp4"]["dispersion_ranges"] = {
            "fixed": rng([x for x in arms if x["arm"].startswith("固定ef")]),
            "reuse": rng([x for x in arms if x["arm"] == "BCAS-复用"]),
            "pool": rng([x for x in arms if x["arm"] == "BCAS-池化"]),
        }

    # ---------- exp5：失败归因 ----------
    p5 = R / "exp5_failure_diag_detail.json"
    if p5.exists():
        d = json.loads(p5.read_text(encoding="utf-8"))
        out["exp5"] = {
            "curve": d.get("curve", []),
            "d3": d.get("d3", []),
            "contrast": d.get("contrast", {}),
            "fail_union": d.get("fail_union", []),
            "fail_inter": d.get("fail_inter", []),
        }

    # ---------- exp6：ZRG 修复验证 ----------
    e6 = read_csv("exp6_zrg.csv")
    if e6:
        rows = []
        for r in e6:
            rows.append({
                "tau": f(r["gain_tau"]),
                "B": int(float(r["预算B"])),
                "arm": r["方案"],
                "recall": f(r["Recall@k 均值"]),
                "interp": f(r["等算力插值召回"]),
                "diff": f(r["召回差(BCAS-插值)"]),
                "p5": f(r["Recall P5"]),
                "zero_recall": f(r["零召回查询数"]),
                "cost": f(r["距离/查询(实测)"]),
                "dispersion": f(r["算力离散度(max/mean)"]),
                "over_budget": f(r["超预算率"]),
            })
        out["exp6"] = {"rows": rows, "tau001": [x for x in rows if x["tau"] == 0.01]}

    # ---------- exp2：邻居选择消融 + eps 扫描 ----------
    e2a = read_csv("exp2a_neighbor_selection.csv")
    if e2a:
        out["exp2a"] = [{
            "config": r["配置"],
            "build_distance": f(r["构图距离计算"]),
            "edges": f(r["边数"]),
            "avg_degree": f(r["平均出度"]),
            "recall": f(r["Recall@k"]),
            "qps": f(r["QPS"]),
            "cost_per_query": f(r["距离计算/次查询"]),
            "build_s": f(r["构建(s)"]),
        } for r in e2a]
    e2b = read_csv("exp2b_early_stop_eps.csv")
    if e2b:
        out["exp2b"] = [{
            "eps": f(r["eps"]),
            "direction": r["终止方向"],
            "recall": f(r["Recall@k"]),
            "delta": f(r["召回相对基线差"]),
            "cost_per_query": f(r["距离计算/次查询"]),
            "saving": f(r["算力相对节省"]),
            "qps": f(r["QPS"]),
        } for r in e2b]

    # ---------- exp3：规模扩展 ----------
    e3 = read_csv("exp3_scalability.csv")
    if e3:
        out["exp3"] = [{
            "n": int(float(r["n"])),
            "build_s": f(r["构图耗时(s)"]),
            "build_distance": f(r["构图距离计算"]),
            "edges": f(r["边数"]),
            "avg_degree": f(r["平均出度"]),
            "graph_mem_mb": f(r['图内存(C++布局,MB)']),
            "recall": f(r["Recall@k"]),
            "qps": f(r["QPS"]),
            "cost_per_query": f(r["距离计算/次查询"]),
            "cost_over_n": f(r["距离比n"]),
        } for r in e3]

    return out


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_md(d: dict) -> str:
    L = ["# 权威数字表（由 results 自动生成，禁止手改）", ""]
    L.append("> 生成脚本：`extract_authoritative.py`。PPT / 报告 / 计划书中的数字必须与本表一致。")
    L.append("")
    c = d["config"]
    L.append(f"统一口径：{c['dataset']}，n={c['n']}，nq={c['nq']}，k={c['k']}，m={c['m']}，efc={c['efc']}，seed={c['seed']}")
    L.append("")

    if "exp1" in d:
        e = d["exp1"]
        L.append("## exp1 · 表 4-1 召回-算力前沿（`results/exp1_n50k.csv`）")
        L.append("")
        L.append(f"构图耗时 **{e['build_s']:.1f} s**")
        L.append("")
        L.append("| 方法 | ef | Recall@10 | QPS | 延迟(ms) | 距离/查询 |")
        L.append("|---|---|---|---|---|---|")
        for r in e["rows"]:
            L.append(f"| {r['method']} | {r['ef']} | {fmt(r['recall'])} | {fmt(r['qps'],1)} | "
                     f"{fmt(r['latency_ms'])} | {fmt(r['cost_per_query'],1)} |")
        L.append("")

    if "exp4" in d:
        e = d["exp4"]
        L.append("## exp4 · 表 4-2 等算力预算对比（`results/exp4_n50k_pool.csv`）")
        L.append("")
        L.append(f"总行数 {e['total_rows']}；超预算率恒 0 的配置 **{e['over_budget_zero_count']}** 组；"
                 f"超预算率非 0 的配置 **{e['over_budget_nonzero_count']}** 组"
                 f"（臂：{', '.join(e['over_budget_nonzero_arms'])}）")
        L.append("")
        L.append(f"算力离散度区间：固定 ef {e['dispersion_ranges']['fixed']}，"
                 f"复用 {e['dispersion_ranges']['reuse']}，池化 {e['dispersion_ranges']['pool']}")
        L.append("")
        for tau_key, label in (("tau001", "τ=0.01（主干）"), ("tau0", "τ=0"), ("tau02", "τ=0.2")):
            L.append(f"### {label}")
            L.append("")
            L.append("| B | 方案 | 对齐ef | 均值召回 | 等算力插值 | 召回差 | P5 | 零召回 | 距离/查询 | 离散度 | 利用率 | 超预算率 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in e[tau_key]:
                L.append(f"| {r['B']} | {r['arm']} | {r['align_ef']} | {fmt(r['recall'])} | {fmt(r['interp'])} | "
                         f"{fmt(r['diff'])} | {fmt(r['p5'])} | {fmt(r['zero_recall'],0)} | {fmt(r['cost'],1)} | "
                         f"{fmt(r['dispersion'],2)} | {fmt(r['util'])} | {fmt(r['over_budget'])} |")
            L.append("")

    if "exp5" in d:
        e = d["exp5"]
        L.append("## exp5 · 失败归因（`results/exp5_failure_diag_detail.json`）")
        L.append("")
        L.append("### 定位实验 1：放大 ef 能否救回失败查询")
        L.append("")
        L.append("| ef | 距离/查询 | 恢复数/样本 | 恢复率 | 失败组 d_ret/d_true |")
        L.append("|---|---|---|---|---|")
        for r in e["d3"]:
            L.append(f"| {r['ef']} | {r['距离/查询']:.0f} | {r['失败组恢复数']:.0f}/{r['失败组样本']:.0f} | "
                     f"{r['失败组恢复率']:.2%} | {r['失败组d_ret/d_true']:.4f} |")
        L.append("")
        ct = e["contrast"]
        if ct:
            L.append(f"### 定位实验 2：近邻对比度\n\n失败组 {ct['fail']:.4f} vs 对照组 {ct['control']:.4f}"
                     f"（差 {abs(ct['fail']/ct['control']-1)*100:.2f}%）\n")
        L.append("### 定位实验 3：零召回组拿到的算力")
        L.append("")
        L.append("| ef | 零召回组算力/全体 | 满分组算力/全体 |")
        L.append("|---|---|---|")
        for r in e["curve"]:
            L.append(f"| {r['ef']} | {r['零召回组算力/全体']:.4f} | {r['满分组算力/全体']:.4f} |")
        L.append("")

    if "exp6" in d:
        e = d["exp6"]
        L.append("## exp6 · ZRG 修复验证（`results/exp6_zrg.csv`，主干 τ=0.01）")
        L.append("")
        L.append("| B | 方案 | 均值召回 | P5 | 零召回 | 距离/查询 | 超预算率 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in e["tau001"]:
            L.append(f"| {r['B']} | {r['arm']} | {fmt(r['recall'])} | {fmt(r['p5'])} | "
                     f"{fmt(r['zero_recall'],0)} | {fmt(r['cost'],1)} | {fmt(r['over_budget'])} |")
        L.append("")

    if "exp2a" in d:
        L.append("## exp2a · 邻居选择消融")
        L.append("")
        L.append("| 配置 | 构图距离 | 边数 | 平均出度 | Recall@10 | 距离/查询 | 构建(s) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in d["exp2a"]:
            L.append(f"| {r['config']} | {r['build_distance']:.0f} | {r['edges']:.0f} | {r['avg_degree']:.2f} | "
                     f"{fmt(r['recall'])} | {fmt(r['cost_per_query'],1)} | {fmt(r['build_s'],1)} |")
        L.append("")

    if "exp2b" in d:
        L.append("## exp2b · eps 双向扫描")
        L.append("")
        L.append("| eps | 方向 | Recall@10 | 相对基线差 | 距离/查询 | 算力节省 |")
        L.append("|---|---|---|---|---|---|")
        for r in d["exp2b"]:
            L.append(f"| {r['eps']} | {r['direction']} | {fmt(r['recall'])} | {fmt(r['delta'])} | "
                     f"{fmt(r['cost_per_query'],1)} | {fmt(r['saving'])} |")
        L.append("")

    if "exp3" in d:
        L.append("## exp3 · 规模扩展")
        L.append("")
        L.append("| n | 构图(s) | 构图距离 | 边数 | 平均出度 | Recall@10 | 距离/查询 | 距离/n |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in d["exp3"]:
            L.append(f"| {r['n']:,} | {fmt(r['build_s'],1)} | {r['build_distance']:.0f} | {r['edges']:.0f} | "
                     f"{r['avg_degree']:.2f} | {fmt(r['recall'])} | {fmt(r['cost_per_query'],1)} | {fmt(r['cost_over_n'])} |")
        L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    d = build()
    (R / "authoritative_numbers.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (R / "authoritative_numbers.md").write_text(render_md(d), encoding="utf-8")
    print("written:", R / "authoritative_numbers.json", R / "authoritative_numbers.md")
