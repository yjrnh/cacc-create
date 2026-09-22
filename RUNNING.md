# 运行手册（自跑用）

按从短到长、从轻到重排列。**所有命令都用绝对路径指向解释器**——直接 `python` 在 Windows 上可能命中不同版本。

## 0. 环境（一次性，已在归档机装好）

| 解释器 | 路径 | 装好的依赖 |
|---|---|---|
| **managed python** (推荐) | `C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe` | numpy 2.4.2 + h5py 3.16.0 |
| system python（仅出图用） | `C:/Python314/python.exe` | matplotlib 3.10.8 |

可选基线 `faiss-cpu` / `hnswlib` 没装，**不影响主链路**——冒烟会跳过它们。

如果你换到新机器需要重装：
```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" -m pip install -r requirements.txt
"C:/Python314/python.exe" -m pip install matplotlib
```

## 1. 冒烟（30 秒，5 步）

```
cd E:\工作\caff\code
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp0_smoke.py
```

预期：5 段 banner，最后一段打印「冒烟通过」。
**通过了**就说明 5 个环节（数据 / 构图 / 搜索 / 计数 / 评测）都正常。

## 2. 基线表（5 分钟，exp1 合成版）

```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp1_recall_qps.py --synthetic --n 20000 --queries 200 --ef 10 40 160
```

输出：`results/exp1_synthetic_n20k.csv` + `.md`
对应报告：**表 4-1**（BCAS vs faiss vs hnswlib 的等 ef 对比）

## 3. 消融（10 分钟，exp2 邻居选择 + ε）

```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp2_ablation.py
```

输出：`results/exp2a_neighbor_selection.csv/json/md` + `results/exp2b_early_stop_eps.csv/json/md`
对应报告：**表 4-4**（A 邻居选择 5 配置 / B ε 双向扫描）

## 4. 规模扩展（10 分钟，exp3 小规模）

```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp3_scalability.py --synthetic --sizes 2000 5000 --queries 100 --k 10 --m 8 --efc 40
```

输出：`results/exp3_scalability.csv/md`
对应报告：**表 4-3**（n=5k/1w/2w/5w 的构图代价与查询算力）

## 5. 主实验（30-45 分钟，exp4 等算力对比）

```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp4_budget.py --synthetic --n 50000 --queries 500 --k 10 --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --ef-grid 10 20 40 80 160 320 --pool --pool-cap-factor 8 --reuse-index --tag exp4_n50k_pool
```

输出：`results/exp4_n50k_pool.{csv,json,md}`
对应报告：**表 4-2**（48 行 BCAS 三臂 vs 固定 ef 基线）
**这是项目的核心实验。**

## 6. 失败归因（30 秒，exp5）

```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp5_failure_diag.py --synthetic --n 50000 --queries 500 --k 10 --m 16 --efc 100 --reuse-index --diag-ef 160 320 640 1024 2048 --tag exp5_failure_diag
```

**必须复用 exp4 的同一张图**（`--reuse-index`），否则不可比。
输出：`results/exp5_failure_diag.{csv,json,md}`
对应报告：**§4.6 第 7 条**（早停反向惩罚的可恢复性证据）

## 7. 修复验证（30-45 分钟，exp6 ZRG）

```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp6_zero_recall_guard.py --synthetic --n 50000 --queries 500 --k 10 --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --pool --pool-cap-factor 8 --reuse-index --tag exp6_zrg
```

输出：`results/exp6_zrg.{csv,json,md}`
对应报告：**§4.6 第 9 条 / 附录 G**（B=1470 池化臂零召回 23→2）

## 8. 出图（10 秒）

```bash
cd E:\工作\caff\code
"C:/Python314/python.exe" experiments/make_figures.py
```

输出：`figures/fig1_frontier.png` + `fig2_ablation.png` + `fig3_bcas.png`

## 9. 转 PDF（30 秒）

```bash
"C:/Users/LENOVO/.workbuddy/binaries/python/envs/default/Scripts/python.exe" experiments/build_report_pdf.py
```

输入：`../.workbuddy/artifacts/CACC技术报告_草稿v1.md`（或你自己的路径）
输出：`CACC技术报告_v1.pdf`

> PDF 生成脚本里路径是相对 `experiments/` 的，会向上找 `.workbuddy/artifacts/`。

---

## 全跑一遍（如果你只想一键）

```bash
cd E:\工作\caff\code
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp0_smoke.py
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp1_recall_qps.py --synthetic --n 20000 --queries 200 --ef 10 40 160
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp2_ablation.py
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp3_scalability.py --synthetic --sizes 2000 5000 10000 20000 --queries 300 --k 10 --m 8 --efc 40
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp4_budget.py --synthetic --n 50000 --queries 500 --k 10 --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --ef-grid 10 20 40 80 160 320 --pool --pool-cap-factor 8 --reuse-index --tag exp4_n50k_pool
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp5_failure_diag.py --synthetic --n 50000 --queries 500 --k 10 --m 16 --efc 100 --reuse-index --diag-ef 160 320 640 1024 2048 --tag exp5_failure_diag
"C:/Users/LENOVO/.workbuddy/binaries/python/versions/3.13.12/python.exe" experiments/exp6_zero_recall_guard.py --synthetic --n 50000 --queries 500 --k 10 --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --pool --pool-cap-factor 8 --reuse-index --tag exp6_zrg
"C:/Python314/python.exe" experiments/make_figures.py
"C:/Users/LENOVO/.workbuddy/binaries/python/envs/default/Scripts/python.exe" experiments/build_report_pdf.py
```

预计总耗时：**1.5–2 小时**（含 n=5 万图的一次性构建 + 缓存复用）。

## 关键参数速查

| 参数 | 含义 | 实验里用 |
|---|---|---|
| `--n` | 数据点总数 | 2000 / 5000 / 50000 |
| `--queries` / `--nq` | 查询数 | 100–500 |
| `--k` | 找最近几个 | 10 |
| `--m` | 每个点连几条边 | 8（小）/ 16（正式） |
| `--efc` / `--ef_construction` | 构图搜索宽度 | 40 / 100 |
| `--ef` / `--ef-grid` | 查询搜索宽度（可多个） | 10, 20, 40, 80, 160, 320 |
| `--budgets` / `--auto-budgets 4` | BCAS 预算档 | auto = 4 档覆盖 B=425 到 1470 |
| `--gain-tau` | 早停阈值 τ | 0.0 / 0.01 / 0.2 |
| `--pool --pool-cap-factor 8` | 池化 + 单条上限倍数 | cap_max = 8×B |
| `--reuse-index` | 复用已缓存的 HNSW 图 | 重要，否则每次重算 |
| `--synthetic` | 用合成数据（绕开 SIFT1M 下载） | 全部实验都用 |

## 输出文件位置

```
results/                       实验数据（CSV / JSON / MD）
figures/                       三张图（PNG）
data/index_cache/              HNSW 索引缓存（PKL + meta.json）
.workbuddy/artifacts/          报告源文件 + PDF
```

## 常见坑

1. **看不到进度** — 每个实验会在关键节点打印 banner（`===`），但**不打印总体进度条**。如果你 5 分钟没新输出，先看 banner 的最后一行。
2. **n=5 万图要 20 分钟** — 第一次跑 exp4 / exp5 / exp6 会构一次图，三者通过 `--reuse-index` 共享。第二次相同参数秒开。
3. **缓存键变了** — 改了 `--m` / `--efc` / 数据源会触发重建，旧缓存保留在 `data/index_cache/`。`du -sh data/index_cache/` 看一下大小。
4. **P1 别用** — 报告里 P5 + 零召回条数才是对的尾部口径（nq=500 时 P1 饱和）。
5. **`--reuse-index` 必须传** — 不传会把所有 m/efc 组合的图都重算一遍。
