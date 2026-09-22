"""把 results/ 里的结果画成报告与 PPT 用的图（图 1 / 图 2 / 图 3）。

    python experiments/make_figures.py

设计取舍（为什么是这几张图）
---------------------------
图 1 分两栏，这是本项目最诚实的一处表达：
  (a) Recall ~ 距离计算/次查询 —— 只有自实现才拿得到这个数，faiss/hnswlib 不暴露；
  (b) Recall ~ QPS —— 在这个尺度上我们落后于高度优化的 C++ 实现，不藏。
把两者并排放，读者一眼能看出"算法层可比、工程常数可比"是两回事。

图 2 展示消融：把"邻居选择策略"这个图索引的核心设计点拆开看，
既给出正结果（heuristic 明显优于 simple），也给出负结果（扩展候选 7 倍构图代价、零收益）。

图 3 用三栏把"均值"与"尾部"分开：固定 ef 的均值能被插值对齐，但尾部不能 ——
这正是 BCAS 存在的理由，也是它相对"简单调大 ef"的增量所在。

⚠ 图 3 的尾部栏**不用 P1**：nq=500 时 np.percentile 的 1% 插值下标落在第 5/6 小值之间，
零召回查询 ≥6 条就把 P1 钉死在 0.0000（与均值、算力无关）。实测 nq=500 下零召回恒有
21–24 条，P1 全表为 0 —— 它在这里是**饱和指标，没有鉴别力**（见技术报告 §4.3 表注 1、§4.6-8）。
因此尾部改用两个**非参数**口径同时报出：
  (b) Recall P5（最差 5% 查询的召回分位）
  (c) 零召回查询数（recall == 0 的查询条数，越低越好）
(c) 是这张图信息量最大的一栏：固定 ef 从 ef=10 到 ef=320 把零召回从 24 条压到 5 条，
而 BCAS 在同等算力下**一条都没压下去**（恒 23–24 条）——这是"早停反向惩罚"最直接的视觉证据。

依赖 matplotlib；中文字体用系统自带的雅黑，缺失时回落到默认字体并告警。
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# 浅色主题：白底、深色字、低饱和配色，直接放进 PDF 不刺眼
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9ced6",
    "axes.labelcolor": "#1f2933",
    "text.color": "#1f2933",
    "xtick.color": "#4b5563",
    "ytick.color": "#4b5563",
    "axes.grid": True,
    "grid.color": "#e6e9ee",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.size": 10,
})

C_OURS = "#1f6feb"
C_BRUTE = "#b42318"
C_FAISS = "#8a63d2"
C_HNSWLIB = "#c26b1f"
C_IVF = "#7a7f87"
C_FIXED = "#7a7f87"
C_BCAS = "#0f7b6c"
C_RESTART = "#c26b1f"
C_POOL = "#8a63d2"


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.DictReader(fh)]


def fnum(v, default=np.nan) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def pick(*cands: str) -> str:
    """取第一个存在的文件；都不存在返回空串。"""
    for c in cands:
        if os.path.exists(os.path.join(RESULTS, c)):
            return os.path.join(RESULTS, c)
    return ""


def thousands(x, _pos=None) -> str:
    return f"{x:,.0f}"


# --------------------------------------------------------------------------
# 图 1：召回-算力前沿
# --------------------------------------------------------------------------
def fig1_frontier() -> str | None:
    src = pick("exp1_n50k.csv", "exp1_recall_qps.csv")
    rows = read_csv(src)
    if not rows:
        print("  [skip] 图1：没有 exp1 结果文件")
        return None
    tag = os.path.basename(src).replace(".csv", "")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # ---- (a) 算法层：召回 ~ 距离计算次数 ----
    ours = [(fnum(r["距离计算/次查询"]), fnum(r["Recall@k"]), r["ef"]) for r in rows
            if r["方法"] == "HNSW(ours)"]
    ours = sorted(x for x in ours if np.isfinite(x[0]))
    if ours:
        xs, ys, efs = zip(*ours)
        ax1.plot(xs, ys, "-o", color=C_OURS, lw=2, ms=6, label="HNSW（自实现）")
        for x, y, ef in ours:
            ax1.annotate(f"ef={ef}", (x, y), textcoords="offset points", xytext=(6, -12),
                         fontsize=8.5, color=C_OURS)

    brute = [(fnum(r["距离计算/次查询"]), fnum(r["Recall@k"])) for r in rows
             if r["方法"] == "BruteForce"]
    brute = [(x, y) for x, y in brute if np.isfinite(x) and np.isfinite(y)]
    if brute:
        bx, by = brute[0]
        ax1.axvline(bx, color=C_BRUTE, ls="--", lw=1.6)
        ax1.plot([bx], [by], "s", color=C_BRUTE, ms=8, label=f"暴力搜索（{bx:,.0f} 次/查询）")
        if ours:
            ax1.annotate("", xy=(bx, by), xytext=(ours[-1][0], ours[-1][1]),
                         arrowprops=dict(arrowstyle="<->", color="#9aa3ad", lw=1.2))
            mid = np.sqrt(bx * ours[-1][0])
            ax1.text(mid, (by + ours[-1][1]) / 2 - 0.02,
                     f"{bx / ours[-1][0]:.1f}×", ha="center", fontsize=10, color="#4b5563")

    ax1.set_xscale("log")
    ax1.set_xlabel("每次查询的距离计算次数（算法层，与硬件无关）")
    ax1.set_ylabel("Recall@10")
    ax1.set_ylim(0.65, 1.02)
    ax1.set_title("(a) 算法层：召回 ~ 距离计算次数", fontsize=11, pad=10)
    ax1.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax1.legend(loc="lower right", fontsize=9)

    # ---- (b) 工程层：召回 ~ QPS（含 C++ 基线） ----
    series = {
        "HNSW（自实现, Python）": (C_OURS, "HNSW(ours)", "-o"),
        "faiss-HNSW（C++）": (C_FAISS, "faiss-HNSW", "-s"),
        "hnswlib（C++）": (C_HNSWLIB, "hnswlib", "-^"),
        "faiss-IVFPQ（C++）": (C_IVF, "faiss-IVFPQ", "-d"),
    }
    for label, (color, key, fmt) in series.items():
        pts = sorted((fnum(r["QPS"]), fnum(r["Recall@k"])) for r in rows if r["方法"] == key)
        pts = [(q, rec) for q, rec in pts if np.isfinite(q) and np.isfinite(rec)]
        if pts:
            qs, rs = zip(*pts)
            ax2.plot(qs, rs, fmt, color=color, lw=1.9, ms=6, label=label)

    bf = [(fnum(r["QPS"]), fnum(r["Recall@k"])) for r in rows if r["方法"] == "BruteForce"]
    bf = [(q, rec) for q, rec in bf if np.isfinite(q)]
    if bf:
        ax2.plot([bf[0][0]], [bf[0][1]], "*", color=C_BRUTE, ms=14, label="暴力搜索")

    ax2.set_xscale("log")
    ax2.set_xlabel("QPS（含 Python 调用开销的保守墙钟口径）")
    ax2.set_ylabel("Recall@10")
    ax2.set_ylim(0.65, 1.02)
    ax2.set_title("(b) 工程层：召回 ~ QPS", fontsize=11, pad=10)
    ax2.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"图 1　召回-算力前沿（数据：{tag}）", fontsize=12.5, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig1_frontier.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 图 2：邻居选择 / 剪枝消融
# --------------------------------------------------------------------------
def fig2_ablation() -> str | None:
    src = pick("exp2a_neighbor_selection.csv")
    rows = read_csv(src)
    if not rows:
        print("  [skip] 图2：没有 exp2a 结果文件")
        return None

    labels = [r["配置"] for r in rows]
    short = [lb.replace("heuristic, ", "").replace("（默认）", "\n(默认)") for lb in labels]
    build = [fnum(r["构图距离计算"]) for r in rows]
    recall = [fnum(r["Recall@k"]) for r in rows]
    qcost = [fnum(r["距离计算/次查询"]) for r in rows]
    deg = [fnum(r["平均出度"]) for r in rows]

    x = np.arange(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    bars = ax1.bar(x, build, color=["#9aa3ad" if i else C_OURS for i in range(len(rows))],
                   width=0.62, edgecolor="#ffffff", linewidth=1.2)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=9)
    ax1.set_ylabel("构图期距离计算次数（对数轴）")
    ax1.set_title("(a) 构图代价", fontsize=11, pad=10)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    base = build[1]                      # heuristic（默认）作为基准
    for b, v in zip(bars, build):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.12, f"×{v/base:.2f}",
                 ha="center", fontsize=9, color="#1f2933")
    ax1.set_ylim(top=max(build) * 3.2)

    ax2b = ax2.twinx()
    ax2.bar(x, recall, color="#cfe0ff", width=0.62, edgecolor=C_OURS, linewidth=1.2,
            label="Recall@10（左轴）")
    ax2b.plot(x, qcost, "-o", color=C_RESTART, lw=2, ms=6, label="距离计算/次查询（右轴）")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=9)
    ax2.set_ylabel("Recall@10")
    ax2b.set_ylabel("查询期距离计算次数", color=C_RESTART)
    ax2b.tick_params(axis="y", colors=C_RESTART)
    ax2.set_ylim(0.0, 1.15)
    ax2b.set_ylim(0.0, max(qcost) * 1.45)
    ax2.set_title("(b) 查询质量与查询代价", fontsize=11, pad=10)
    for i, (rc, qc) in enumerate(zip(recall, qcost)):
        ax2.text(i, rc + 0.03, f"{rc:.3f}", ha="center", fontsize=9)
    h2, l2 = ax2b.get_legend_handles_labels()
    h1, l1 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)

    # 参数不再写死在标题里：消融可能换 n / m / ef，写死会出现"标题与数据不符"。
    # 真正的参数值由 exp2 输出的 md 表头承载，报告里引图时对照表头即可。
    fig.suptitle("图 2　图结构消融：邻居选择策略与剪枝（参数见 results/exp2_ablation.md 表头）",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_ablation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 图 3：BCAS 与固定 ef 的等算力对比
# --------------------------------------------------------------------------
def _parse_budget_key(key):
    """exp4 的 budgets 键有两种历史形态：旧的 `300`（纯预算）与新的 `tau0.01_B300`。

    新形态是多 tau 敏感性分析引入的（固定 ef 曲线只扫一次，多个 tau 共用）。
    这里统一解析成 (tau, B)；旧形态的 tau 记为 None。
    """
    if "_B" in key:
        tau_s, b_s = key.rsplit("_B", 1)
        try:
            return float(tau_s.replace("tau", "")), int(b_s)
        except ValueError:
            return None, None
    try:
        return None, int(key)
    except ValueError:
        return None, None


def _zero_count(arm_stats: dict) -> float:
    """从逐查询召回里数出"零召回查询"的条数。

    为什么不用结果里已有的分位数：P1/P5 这些分位点在 nq=500 时会饱和（见文件头说明），
    而"有多少条查询一条近邻都没找回来"是完全非参数的，不受分位点插值规则影响。
    """
    rq = arm_stats.get("recalls_perquery")
    if not rq:
        return np.nan
    return float(sum(1 for r in rq if r <= 0.0))


def fig3_bcas() -> str | None:
    # 优先用正式 run 的池化版结果（n=5 万 / nq=500 / 含三条臂）：这是报告表 4-2 的数据来源。
    # 回退顺序保留旧文件，是为了脚本在只有冒烟结果时仍能出图，不至于整脚本报错。
    src = pick("exp4_n50k_pool.json", "exp4_budget.json", "exp4_smoke.json")
    if not src:
        print("  [skip] 图3：没有 exp4 结果文件")
        return None
    with open(src, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    curve = d["detail"]["curve"]
    raw = d["detail"]["budgets"]
    tag = os.path.basename(src).replace(".json", "")

    # 多 tau 时只画一个 tau，否则几条曲线会重叠成一团、图反而说不清。
    # 选 0.01（推荐工作点）；没有 0.01 就选最接近 0 的那个，并在图注里标明用的是哪个。
    parsed = []
    for k, v in raw.items():
        tau, b = _parse_budget_key(k)
        if b is not None:
            parsed.append((tau, b, v))
    taus = sorted({t for t, _, _ in parsed if t is not None})
    if taus:
        tau_used = min(taus, key=lambda t: abs(t - 0.01))
        parsed = [(t, b, v) for t, b, v in parsed if t == tau_used]
        note = f"gain_tau={tau_used:g}"
    else:
        note = "gain_tau=单一"
    parsed = sorted(parsed, key=lambda t: t[1])

    # 两套命名必须显式映射：curve 用短键（p5/zero_recall），逐预算条目用 perquery_eval 的
    # 原始键（recall_p5），零召回数则库里没存、要现算（_zero_count）。不映射就 KeyError。
    CURVE_KEY = {"recall_mean": "recall_mean", "p5": "p5", "zero": "zero_recall"}
    ARM_KEY = {"recall_mean": "recall_mean", "p5": "recall_p5"}

    # 三条臂都画：池化臂是"再分配确实发生"的证据，复用臂是默认形态，重算臂是对照。
    # 只画一条臂会让图与报告表 4-2 对不上（表里三臂的结果方向不同，这正是本章的重点）。
    arms = (("BCAS-池化", C_POOL, "-s", 6.5, "BCAS 批次共享预算池"),
            ("BCAS-复用", C_BCAS, "-o", 6.0, "BCAS 复用索引（默认）"),
            ("BCAS-重算", C_RESTART, "--^", 5.5, "BCAS 重算（对照臂）"))

    fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.9))
    panels = ((axes[0], "recall_mean", "(a) 均值：Recall@10"),
              (axes[1], "p5", "(b) 尾部：Recall P5（最差 5% 查询）"),
              (axes[2], "zero", "(c) 尾部：零召回查询数（越少越好）"))

    for pane_i, (ax, key, title) in enumerate(panels):
        # ---- 固定 ef 的等算力基准曲线 ----
        pts = [(c["cost"], c.get(CURVE_KEY[key]), c.get("ef")) for c in curve]
        pts = [(x, y, e) for x, y, e in pts if y is not None and np.isfinite(y)]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=C_FIXED,
                    lw=2.0, ms=6.5, label="固定 ef（等算力基准曲线）")
            for x, y, e in pts:
                ax.annotate(f"ef={e}", (x, y), textcoords="offset points", xytext=(5, -13),
                            fontsize=8.5, color=C_FIXED)

        # ---- 三条 BCAS 臂（按实测距离计算次数排序）----
        for arm, color, fmt, ms, label in arms:
            ap = []
            for _tau, b, info in parsed:
                st = info.get(arm)
                if not st:
                    continue
                y = _zero_count(st) if key == "zero" else st.get(ARM_KEY[key])
                if y is None or not np.isfinite(y):
                    continue
                ap.append((st["dists_mean"], float(y), b))
            ap.sort(key=lambda t: t[0])
            if not ap:
                continue
            ax.plot([p[0] for p in ap], [p[1] for p in ap], fmt, color=color, lw=1.7, ms=ms,
                    label=label if pane_i == 0 else None)
            # 只在池化臂上标预算值：三臂在同一预算下算力不同，全标会糊成一团。
            if arm == "BCAS-池化":
                for x, y, b in ap:
                    ax.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(6, 6),
                                fontsize=8.5, color=color)

        ax.set_xscale("log")
        ax.set_xlabel("每次查询的实测距离计算次数（对数轴）")
        ax.set_title(title, fontsize=11, pad=10)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))

        if key == "zero":
            ax.set_ylabel("零召回查询数 / 共 500")
            ax.set_ylim(0.0, 27.5)
            ax.axhline(24.0, color=C_FIXED, ls=":", lw=1.2)
            ax.text(0.97, 0.93, "固定 ef=10 的水平（24 条）", ha="right", va="center",
                    transform=ax.transAxes, fontsize=8.5, color=C_FIXED)
        elif key == "p5":
            ax.set_ylabel("Recall@10（P5）")
            ax.set_ylim(0.05, 1.06)
        else:
            ax.set_ylabel("Recall@10（均值）")
            ax.set_ylim(0.66, 1.03)

        if pane_i == 0:
            ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(f"图 3　BCAS 与固定 ef 的等算力对比（数据：{tag}，{note}，nq=500）",
                 fontsize=12.5, y=1.0)
    fig.text(0.5, -0.055,
             "尾部两栏（b)(c) 为非参数口径；(c) 越低越好。P1 在 nq=500 下饱和为 0.0000、无鉴别力，"
             "本图不用作判据（技术报告 §4.3 表注 1 / §4.6-8）。",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_bcas.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print("生成图 1 / 2 / 3 …")
    for fn in (fig1_frontier, fig2_ablation, fig3_bcas):
        out = fn()
        if out:
            print(f"  ✓ {os.path.relpath(out, ROOT)}  ({os.path.getsize(out)/1024:.0f} KB)")
    print("\n注意：图 1/3 的曲线形状依赖实际跑出的数据；数据更新后重跑本脚本即可。")


if __name__ == "__main__":
    main()
