# CACC-BCAS：预算可控的稠密向量近似搜索

> 参赛仓库：https://github.com/yjrnh/cacc-create

自实现近邻图（HNSW）索引 + 预算约束自适应检索（BCAS），配一套可复现的评测装置。
纯 CPU，不依赖 GPU，无需下载大模型。

## 一句话

在同样的召回率下，让近邻检索算得更少；并且在**给定的算力预算**下，让最差那批查询的
召回率不再崩坏。衡量的尺子是**距离计算次数**，一个与机器性能无关的纯算法层指标。

## 两层增量（这是项目区别于"复现 HNSW"的地方）

| 层 | 内容 | 文件 |
|---|---|---|
| 索引层 | 自实现 HNSW：邻居选择启发式可插拔（Algorithm 3/4）、(1+eps) 提前终止、全链路距离计数 | `src/hnsw.py` |
| **调度层** | **BCAS：迭代加宽 + 硬预算守卫 + 边际收益早停**，把固定 ef 换成"以距离计算次数为预算单位"的自适应搜索 | `src/adaptive.py` |

调度层要解决的工程问题：线上检索有硬性算力 SLA，而 `ef` 是图搜索宽度、不是算力，
调参者无法据此承诺任何东西；同时固定 ef 把预算平均分给了难度天差地别的查询，
导致召回率均值好看、尾部崩坏。

## 快速开始

```bash
# 1) 环境（已在本机 venv 中装好 numpy / h5py / faiss-cpu / hnswlib）
pip install -r requirements.txt

# 2) 冒烟：30 秒验证五个环节全部正常
python experiments/exp0_smoke.py

# 3) 主实验（无下载，合成数据，约 5 分钟）
python experiments/exp1_recall_qps.py --synthetic --n 20000 --queries 200 --ef 10 40 160

# 4) 正式实验（SIFT1M 子集，首次会下载约 500MB 到 data/）
python experiments/exp1_recall_qps.py --n 50000 --queries 1000

# 5) 预算自适应实验（本项目的增量所在）
python experiments/exp4_budget.py --synthetic --n 20000 --queries 300 \
    --budgets 300 600 1200 2400 --ef-grid 10 20 40 80 160 320
```

## 目录

```
src/
  data.py        数据：合成数据（带簇，自带真值）/ ann-benchmarks HDF5 / 子采样重算真值
  hnsw.py        自实现 HNSW：邻居选择可插拔、(1+eps) 提前终止、距离计数、持久化
  adaptive.py    BCAS：可恢复搜索状态、硬预算守卫、边际收益早停、逐查询评测
  baselines.py   基线：BruteForce / faiss-HNSW / faiss-IVFPQ / hnswlib
  bench.py       统一评测：Recall@k、QPS、距离计算次数、Pareto 前沿
  metrics.py     指标定义
  counter.py     距离计算计数器
experiments/
  exp0_smoke.py           冒烟测试（必跑）
  exp1_recall_qps.py      主实验：Recall–QPS 前沿
  exp2_ablation.py        消融：邻居选择策略 + (1+eps) 提前终止
  exp3_scalability.py     规模：n 从 1 万到 10 万
  exp4_budget.py          等算力预算对比：固定 ef vs BCAS（报告 4.2 节）
results/                  每次实验输出 CSV + JSON + Markdown 表
data/                     数据集缓存 + index_cache/（构图很慢，务必复用）
```

## 指标口径（写死在报告里）

| 指标 | 定义 | 为什么重要 |
|---|---|---|
| Recall@k | 逐查询求 \|返回 ∩ 真值\| / k 再平均 | 检索质量，与基线同口径 |
| QPS | 查询条数 / 总墙钟时间 | 工程速度，受语言与实现常数影响 |
| **距离计算/次查询** | 平均每次查询触发的距离计算次数 | **与硬件无关的算法层指标，本项目主指标** |
| 构图距离计算 | 建索引全程的距离计算总数 | 建索引代价的算法层度量 |
| 平均出度 / 边数 | 图结构统计 | 内存代价，与语言无关 |

## 四条必须写进报告的诚实声明

1. **本实现是 Python 原型，QPS 常数因子比 C++ 实现（faiss / hnswlib）高约两个数量级。**
   因此 QPS 一栏只用于验证「距离计算次数下降 → 速度上升」的一致性，不作为算法效率的结论依据。
   算法效率一律以**距离计算次数**为准。

2. **在小规模底库（约 n < 5 万）上，numpy 向量化的暴力搜索 QPS 可能高于 Python 循环的图索引。**
   这是实现常数问题，不是算法问题。图索引的优势体现在**距离计算次数**上，且从 n=3000 起就已成立
   （冒烟测试：图索引 136 次/查询 vs 暴力 3000 次/查询）。正式实验必须取 n ≥ 5 万。

3. **不声称超过 faiss / hnswlib / DiskANN / ScaNN。** 那些是工业级多年打磨的成果。
   本项目的定位是：在同一召回水平下，给出**可解释的邻居选择策略对比**与**可计量的算法层收益**。

4. **BCAS 的"可恢复状态"是近似的，召回折损不隐藏。**
   上一轮把距离劣于当时最差者的候选丢弃了，加宽后不会回头捡。因此 BCAS-复用 的召回率
   可能略低于"每轮从零重算"（BCAS-重算）。`exp4` 把 BCAS-重算 一并报出作为召回上界。
   另一件事同样不隐藏：**BCAS-重算 在结构上无法守住硬预算**（单次固定 ef 的开销事前不可知），
   它的"超预算率"必然非零 —— 这正是"可恢复状态"不是可选优化的原因。

## 报告里可用的三张核心图

- 图 1：Recall@k（横轴）— 距离计算次数/查询（纵轴，对数轴），三到四条曲线（暴力 / faiss-HNSW / 本实现不同配置）。
- 图 2：Recall@k — QPS，用于验证结论一致性，并明确标注常数因子说明。
- 图 3：BCAS vs 固定 ef —— 左轴平均召回率、右轴 **P1 召回率**，横轴为算力预算。
  P1 曲线分叉是"预算再分配"主张的直接证据；若不分叉，就如实说该方法无效。

## 常用命令

```bash
# 消融（邻居选择 5 种配置 + eps 扫描）
python experiments/exp2_ablation.py --synthetic --n 20000 --queries 300

# 规模（看距离计算次数是否随 log n 增长 —— 图索引的根本优势）
python experiments/exp3_scalability.py --synthetic --sizes 5000 10000 20000

# 换数据集
python experiments/exp1_recall_qps.py --dataset glove-100-angular --n 50000

# 预算自适应（复用已构图的索引，别重复构）
python experiments/exp4_budget.py --synthetic --n 50000 --queries 500 \
    --budgets 300 600 1200 2400 4800 --reuse-index
```

## 已知限制

- 构图是纯 Python 循环，n=5 万约需 10–15 分钟；n=10 万约 25–30 分钟。构图后请用 `HNSW.save()` 缓存，别重复构。
  `exp4` 已内置 `data/index_cache/` 复用（缓存键含 n / m / ef_construction，改了任一就重构）。
- 幂等性：固定 `seed=0` 后结果完全可复现。
- 目前只实现 L2 距离；内积/余弦需先归一化并改用对应度量。
- BCAS 的 `gain_tau` 目前是手工阈值，报告需给出它的敏感性分析（换 3 个取值）。
- BCAS 的预算利用率在大预算下会明显低于 1（实测 0.37），这不是浪费：边际收益早停
  已经判定该查询收敛，继续花预算不会提升召回。报告里必须把"利用率低"解释成收敛而非节约失败。
