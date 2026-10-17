"""把 authoritative_numbers.json 渲染成技术报告所需的 Markdown 表格片段。

只负责"生成表格文本"，不直接改报告 —— 由人（或脚本）把它贴进锚点位置，
避免正则误伤正文。
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).parent / "results"
D = json.loads((R / "authoritative_numbers.json").read_text(encoding="utf-8"))


def s(v, nd=4, sign=False):
    if v is None:
        return "—"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        txt = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
        return txt
    return str(v)


def table_4_1() -> str:
    rows = ["| 方法 | ef | Recall@10 | QPS | 延迟(ms) | **距离计算/次查询** | 构建(s) |",
            "|---|---|---|---|---|---|---|"]
    for r in D["exp1"]["rows"]:
        name = {"HNSW(ours)": "HNSW(本文)", "BruteForce": "暴力搜索（numpy）",
                "faiss-HNSW": "faiss-HNSW", "faiss-IVFPQ": "faiss-IVF-PQ",
                "hnswlib": "hnswlib"}.get(r["method"], r["method"])
        cost = "—（库未暴露）" if r["cost_per_query"] is None else f"**{r['cost_per_query']:.1f}**"
        if r["method"] == "HNSW(ours)":
            cost = f"**{r['cost_per_query']:.1f}**"
        if r["method"] == "BruteForce":
            cost = f"**{r['cost_per_query']:.1f}**"
            r = {**r, "ef": "—"}
        recall = f"**{r['recall']:.4f}**" if r["method"] == "HNSW(ours)" else f"{r['recall']:.4f}"
        rows.append(f"| {name} | {r['ef']} | {recall} | {r['qps']:.2f} | {r['latency_ms']:.4f} | {cost} | {r['build_s']:.2f} |")
    return "\n".join(rows)


def table_4_2_main() -> str:
    rows = ["| B | 方案 | 对齐 ef | 有效性 | Recall@10 均值 | 等算力插值召回 | 召回差 | Recall P5 | 零召回数 | 距离/查询 | 算力离散度 | 利用率 | 超预算率 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in D["exp4"]["tau001"]:
        arm = r["arm"]
        bold = arm == "BCAS-池化"
        arm_s = f"**{arm}**" if bold else arm
        diff = s(r["diff"], 4, sign=True)
        if r["valid"] != "有效":
            diff = "-"
        ob = "—" if r["over_budget"] is None else s(r["over_budget"])
        if r["over_budget"] not in (None, 0.0):
            ob = f"**{ob}**"
        disp = s(r["dispersion"], 2)
        if bold:
            disp = f"**{disp}**"
        rows.append(f"| {r['B']} | {arm_s} | {r['align_ef']} | {r['valid']} | {s(r['recall'])} | "
                    f"{s(r['interp'])} | {diff} | {s(r['p5'])} | {s(r['zero_recall'],0)} | "
                    f"{s(r['cost'],1)} | {disp} | {s(r['util'],3)} | {ob} |")
    return "\n".join(rows)


def appendix_e() -> str:
    rows = ["| tau | B | 方案 | 对齐ef | Recall均值 | 等算力插值 | 召回差 | P1 | P5 | 零召回数 | 距离/查询 | 离散度 | 利用率 | 超预算率 | 平均轮次 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in D["exp4"]["rows"]:
        ob = "—" if r["over_budget"] is None else s(r["over_budget"])
        rows.append(f"| {r['tau']} | {r['B']} | {r['arm']} | {r['align_ef']} | {s(r['recall'])} | "
                    f"{s(r['interp'])} | {s(r['diff'],4,True)} | 0 | {s(r['p5'])} | {s(r['zero_recall'],0)} | "
                    f"{s(r['cost'],1)} | {s(r['dispersion'],3)} | {s(r['util'],3)} | {ob} | — |")
    return "\n".join(rows)


def table_4_3() -> str:
    rows = ["| n | 构图耗时(s) | 构图距离计算 | 构图距离计算/n | 边数 | 平均出度 | 图内存(C++布局,MB) | Recall@10 | QPS | **距离计算/次查询** | 距离计算/n |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in D["exp3"]:
        rows.append(f"| {r['n']:,} | {r['build_s']:.1f} | {r['build_distance']:,.0f} | {r['build_distance']/r['n']:,.0f} | "
                    f"{r['edges']:,.0f} | {r['avg_degree']:.2f} | {r['graph_mem_mb']:.2f} | {s(r['recall'])} | "
                    f"{r['qps']:.1f} | **{r['cost_per_query']:.1f}** | {s(r['cost_over_n'])} |")
    return "\n".join(rows)


def table_4_4a() -> str:
    rows = ["| 配置 | 邻居选择 | extend_candidates | keep_pruned | 构图距离计算 | 边数 | Recall@10 | 距离计算/次查询 | 构建(s) |",
            "|---|---|---|---|---|---|---|---|---|"]
    label = {"simple": "A", "heuristic（默认）": "B", "heuristic, 不补足剪枝": "D",
             "heuristic, 扩展候选": "C", "heuristic, 扩展+不补足": "E"}
    sel = {"simple": "simple (Alg.3)", "heuristic（默认）": "heuristic (Alg.4)",
           "heuristic, 不补足剪枝": "heuristic (Alg.4)", "heuristic, 扩展候选": "heuristic (Alg.4)",
           "heuristic, 扩展+不补足": "heuristic (Alg.4)"}
    flags = {"simple": ("-", "-"), "heuristic（默认）": ("False", "True"),
             "heuristic, 不补足剪枝": ("False", "False"), "heuristic, 扩展候选": ("True", "True"),
             "heuristic, 扩展+不补足": ("True", "False")}
    for r in D["exp2a"]:
        c = r["config"]
        ec, kp = flags.get(c, ("-", "-"))
        rows.append(f"| {label.get(c,'?')} | {sel.get(c,c)} | {ec} | {kp} | {r['build_distance']:,.0f} | "
                    f"{r['edges']:,.0f} | {s(r['recall'])} | {s(r['cost_per_query'],1)} | {s(r['build_s'],1)} |")
    return "\n".join(rows)


def table_4_4b() -> str:
    rows = ["| eps | 终止方向 | Recall@10 | 召回相对基线差 | 距离计算/次查询 | 算力相对节省 | QPS |",
            "|---|---|---|---|---|---|---|"]
    for r in D["exp2b"]:
        rows.append(f"| {r['eps']:.4f} | {r['direction']} | {s(r['recall'])} | {s(r['delta'],4,True)} | "
                    f"{s(r['cost_per_query'],2)} | {s(r['saving'])} | {s(r['qps'],1)} |")
    return "\n".join(rows)


if __name__ == "__main__":
    out = {
        "table_4_1": table_4_1(),
        "table_4_2_main": table_4_2_main(),
        "appendix_e": appendix_e(),
        "table_4_3": table_4_3(),
        "table_4_4a": table_4_4a(),
        "table_4_4b": table_4_4b(),
    }
    (R / "_tables_generated.md").write_text(
        "\n\n".join(f"<!-- {k} -->\n{v}" for k, v in out.items()), encoding="utf-8")
    print("written results/_tables_generated.md")
    for k, v in out.items():
        print(f"\n### {k}\n{v}")
