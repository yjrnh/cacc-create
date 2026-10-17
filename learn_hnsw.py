"""HNSW 最小学习脚本 —— 用「跳表的视角」看清 HNSW 的分层结构。

跑法（必须在项目根目录！）：
    python learn_hnsw.py
"""

import numpy as np

from src.hnsw import HNSW          # ← 关键：从根目录用包的方式导入

np.random.seed(0)

# ---------- 1. 造一小批 2 维数据（方便想象成平面上的点）----------
dim = 2
n = 24
X = np.random.rand(n, dim).astype(np.float32)

# ---------- 2. 建索引 ----------
# 注意 m=4（而不是默认的 16）：m 越小，分到高层的节点比例越大，方便观察分层
idx = HNSW(dim=dim, m=4, ef_construction=50, ef_search=30, seed=1)
idx.add_items(X)


def show(index, title):
    """把 HNSW 的每一层打印出来 —— 和跳表的 show() 是同一个视角"""
    print(f"\n{title}")
    print("=" * 60)
    for l in range(len(index.links) - 1, -1, -1):
        nodes = sorted(index.links[l].keys())
        print(f"\n  第{l}层：共 {len(nodes)} 个节点  {nodes}")
        for nd in nodes:
            print(f"      {nd:>2} → {index.links[l][nd]}")
    print(f"\n  入口点(entry_point) = {index.entry_point}   最高层 = {index.max_level}")


show(idx, "【HNSW 的分层结构】每层是一个「图」，不是一条「链」")

# ---------- 3. 查一次看看 ----------
q = np.array([0.5, 0.5], dtype=np.float32)

# 暴力算真实答案，用来对照
d_all = ((X - q) ** 2).sum(axis=1)
truth = np.argsort(d_all)[:5]

print("\n" + "=" * 60)
print("【查询演示】查询点 = [0.5, 0.5]")
print(f"  真实最近的 5 个(暴力穷举): {list(truth)}")
print(f"     它们的平方距离:        {[round(float(d_all[i]), 4) for i in truth]}")

ids, dst = idx.search(q, k=5)
print(f"\n  HNSW 返回的 5 个:         {list(ids)}")
print(f"     它算出的平方距离:      {[round(float(d), 4) for d in dst]}")

hit = len(set(ids.tolist()) & set(truth.tolist()))
print(f"\n  命中 {hit}/5  → 召回率 = {hit / 5:.0%}")

# ---------- 4. 代价 ----------
print("\n" + "=" * 60)
print("【算法层代价】（与硬件无关的指标）")
print(f"  建图总共的距离计算次数: {idx.counter.count:,} 次")
print(f"  暴力搜一次需要:         {n} 次")
print(f"  这次查询新增:           {idx.counter.count:,} → 见下方增量")

before = idx.counter.count
idx.search(q, k=5)
print(f"  单次查询的距离计算:     {idx.counter.count - before} 次  (暴力需 {n} 次)")

print("\n" + idx.memory_report())
