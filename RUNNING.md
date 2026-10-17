# 运行手册（定稿版 · 2026-10-17 重跑校准）

> 本手册的命令与 `run_all_1017.sh` **逐字一致**，是本项目所有报告 / PPT 数字的那一次运行所用的命令。
> 实测总耗时 **约 90 分钟**（单机 CPU，12 逻辑核，无 GPU；n=5 万图缓存需先构建一次）。

## 0. 环境

```bash
cd E:/工作/caff/code_v2
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

| 依赖 | 版本（本机实测） |
|---|---|
| Python | 3.13.12 |
| numpy | 2.4.2 |
| faiss-cpu | 1.15.0（仅基线对比用，缺失时会自动跳过） |
| hnswlib | 已装（同上） |
| h5py / matplotlib | 3.16.0 / 3.11.2（matplotlib 仅出图） |

**为什么必须用 venv 里的解释器**：基线与出图依赖装在这个 venv 里；系统 `python` 可能缺包。
若换机器，`python -m venv .venv` 后按上面装依赖即可。

## 1. 一键全跑

```bash
cd E:/工作/caff/code_v2
bash run_all_1017.sh          # 全流程，日志逐实验写入 logs/
cat logs/_summary.txt         # 每步 rc 与起止时间
```

**实测耗时（2026-10-17 那次）**：

| 步骤 | 墙钟 | 备注 |
|---|---|---|
| exp0 冒烟 | 17 s | 5 环节自检 |
| exp1 基线表 | ≈ 27 min | 其中 n=5 万图构建 922 s（一次性） |
| exp4 主实验 | 32 s | 复用 exp1 的图 |
| exp5 失败归因 | 37 s | 必须复用同一张图 |
| exp6 ZRG 修复 | 36 s | 同上 |
| exp3 规模扩展 | 10 min | 三个规模依次建图 |
| exp2 消融 | 49 min | 5 个配置**并行**构图，墙钟 ≈ 最慢一项（2,932 s） |
| 出图 | 6 s | 三张 PNG |

## 2. 分步命令（与一键脚本同参）

### 2.1 冒烟（17 秒）

```bash
.venv/Scripts/python.exe experiments/exp0_smoke.py
```

预期打印 5 段 banner，最后「冒烟通过」。不通过不要往下跑。

### 2.2 基线表（≈ 27 分钟）→ 报告表 4-1、图 1

```bash
.venv/Scripts/python.exe experiments/exp1_recall_qps.py --synthetic --n 50000 --queries 500 \
  --k 10 --ef 10 20 40 80 160 --m 16 --efc 100 --tag exp1_n50k
```

输出：`results/exp1_n50k.{csv,json,md}`
**ef=40 定标点：Recall 0.9306、距离计算 792.35 次/查询**（本手册用它做缓存/口径自检）。

### 2.3 主实验等算力对比（32 秒，需 2.2 的图）→ 报告表 4-2、附录 E/F、图 3

```bash
.venv/Scripts/python.exe experiments/exp4_budget.py --synthetic --n 50000 --queries 500 \
  --k 10 --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --ef-grid 10 20 40 80 160 320 \
  --pool --pool-cap-factor 8 --reuse-index --tag exp4_n50k_pool
```

输出：`results/exp4_n50k_pool.{csv,json,md}`。**这是项目的核心实验。**
`--reuse-index` 必须传，否则会把每档 ef 的图重算一遍。

### 2.4 失败归因（37 秒）→ 报告 §4.6

```bash
.venv/Scripts/python.exe experiments/exp5_failure_diag.py --synthetic --n 50000 --queries 500 \
  --k 10 --m 16 --efc 100 --reuse-index --diag-ef 160 320 640 1024 2048 --tag exp5_failure_diag
```

输出：`results/exp5_failure_diag.{csv,json,md}` + `exp5_failure_diag_detail.json`（逐查询明细）。
**必须与 exp4 复用同一张图**，否则不可比。

### 2.5 ZRG 修复验证（36 秒）→ 报告 §4.6 / 附录 G、图 3

```bash
.venv/Scripts/python.exe experiments/exp6_zero_recall_guard.py --synthetic --n 50000 --queries 500 \
  --k 10 --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --ef-grid 10 20 40 80 160 320 \
  --pool --pool-cap-factor 8 --reuse-index --tag exp6_zrg
```

输出：`results/exp6_zrg.{csv,json,md}`
定标点（τ=0.01, B=1470, 池化臂）：零召回 **23 → 2**、P5 **0.77 → 0.90**、均值 **0.9424 → 0.9844**、
算力 **1009.4 → 1145.2**（仍 < 1470）。

### 2.6 规模扩展（10 分钟）→ 报告表 4-3

```bash
.venv/Scripts/python.exe experiments/exp3_scalability.py --synthetic --sizes 5000 10000 20000 --queries 500
```

输出：`results/exp3_scalability.{csv,json,md}`
一级指标：距离计算/次查询 **221.5 → 368.3 → 630.7**（占 n 比例 4.43% → 3.68% → 3.15%，单调下降）。

> **QPS 不要跨规模比**：QPS 是单次墙钟测量，三档依次运行、机器负载不同，5,000 档反而高于 10,000 档。

### 2.7 消融（49 分钟）→ 报告表 4-4、图 2

```bash
.venv/Scripts/python.exe experiments/exp2_ablation.py --synthetic --n 20000 --queries 300 \
  --k 10 --ef 80 --m 16 --efc 100
```

输出：`results/exp2a_neighbor_selection.{csv,json}`、`exp2b_early_stop_eps.{csv,json}`、`exp2_ablation.md`
两个定标点：simple **0.6183** vs heuristic **1.0000**（A）；ε=−0.25 时召回崩到 **0.0637**（B）。

> **QPS / 构建秒数的读法**：5 个配置由同一进程池**并行**构图（墙钟 ≈ 最慢一项），
> 因此绝对值偏高、且受资源竞争影响，**不可与 2.2 的串行数据横比**；
> 可横比的是同一批内的比例（如 heuristic / simple 构建 ×5.7）。
> 一级指标是**距离计算/次查询**，该列与并行无关，重跑逐位一致。

### 2.8 出图（6 秒）→ 图 1/2/3

```bash
.venv/Scripts/python.exe experiments/make_figures.py
```

输出：`figures/fig1_frontier.png`、`fig2_ablation.png`、`fig3_bcas.png`

### 2.9 生成权威数字表（1 秒）

```bash
.venv/Scripts/python.exe extract_authoritative.py
```

输出：`results/authoritative_numbers.{json,md}`
**报告 / PPT / 计划书里的每一个实验数字都以它为准**，改数据后必须重跑这一步。

## 3. 关键参数速查

| 参数 | 含义 | 本次用值 |
|---|---|---|
| `--n` | 数据点总数 | 50,000（主链）/ 20,000（消融）/ 5k·1w·2w（扩展） |
| `--queries` | 查询数 | 500（主链）/ 300（消融） |
| `--k` | 最近邻居数 | 10 |
| `--m` | 每点连边数 | 16 |
| `--efc` | 构图搜索宽度 | 100 |
| `--ef` / `--ef-grid` | 查询搜索宽度 | 10 20 40 80 160（基线）+ 320（主实验网格） |
| `--auto-budgets 4` | BCAS 预算档 | 自动取 B = 425 / 643 / 972 / 1470 |
| `--gain-tau` | 早停阈值 τ | 0.0 / 0.01 / 0.2（0.01 为主干） |
| `--pool --pool-cap-factor 8` | 池化 + 单条上限 | cap_max = 8×B |
| `--reuse-index` | 复用已缓存图 | 主链必传 |
| `--synthetic` | 合成簇结构数据 | 全部实验都用（SIFT1M 已下载、未跑全量） |

## 4. 输出位置

```
results/                实验数据（CSV / JSON / MD）+ authoritative_numbers
figures/                三张图（PNG）
data/index_cache/       HNSW 图缓存（PKL + meta.json，含底库 sha1 指纹）
logs/                   每次运行的逐实验日志与 _summary.txt
```

## 5. 常见坑

1. **看不到进度** —— 每个实验只在关键节点打 banner，没有进度条。5 分钟无输出先看 banner 末行。
2. **n=5 万图要 15 分钟** —— 第一次跑 exp1/exp4 会建图，exp4/exp5/exp6 通过 `--reuse-index` 共享。
3. **缓存指纹不匹配会直接报错**（这是特性，不是 bug）—— 改了 `--m`/`--efc`/数据源/nq 会重建；
   若复用了错配的图，旧版本会**静默**返回崩坏的召回（0.9 → 0.02 且不报错），现在会拒绝。
4. **尾部指标用 P5 + 零召回条数**，别用 P1（nq=500 时 P1 饱和，几乎看不出差异）。
5. **QPS 只作方向参考** —— 跨实验、跨机器、并行/串行都不可横比；写进结论的只能是距离计算次数。
