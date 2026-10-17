#!/usr/bin/env bash
# 统一重跑：严格按 CACC技术报告_草稿v1.md 附录 A 的命令
set -u
cd "E:/工作/caff/code_v2" || exit 1
PY=".venv/Scripts/python.exe"
mkdir -p logs
SUM="logs/_summary.txt"
: > "$SUM"

run() {
  local name="$1"; shift
  local t0=$(date +%H:%M:%S)
  echo "===== START $name at $t0"
  "$PY" "$@" > "logs/$name.log" 2>&1
  local rc=$?
  local t1=$(date +%H:%M:%S)
  echo "===== END   $name rc=$rc  $t0 -> $t1"
  echo "$name rc=$rc $t0 $t1" >> "$SUM"
}

run exp0 experiments/exp0_smoke.py

run exp1 experiments/exp1_recall_qps.py --synthetic --n 50000 --queries 500 --k 10 \
    --ef 10 20 40 80 160 --m 16 --efc 100 --tag exp1_n50k

run exp4 experiments/exp4_budget.py --synthetic --n 50000 --queries 500 --k 10 \
    --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --ef-grid 10 20 40 80 160 320 \
    --pool --pool-cap-factor 8 --reuse-index --tag exp4_n50k_pool

run exp5 experiments/exp5_failure_diag.py --synthetic --n 50000 --queries 500 --k 10 \
    --m 16 --efc 100 --reuse-index --diag-ef 160 320 640 1024 2048 --tag exp5_failure_diag

run exp6 experiments/exp6_zero_recall_guard.py --synthetic --n 50000 --queries 500 --k 10 \
    --auto-budgets 4 --gain-tau 0.0 0.01 0.2 --ef-grid 10 20 40 80 160 320 \
    --pool --pool-cap-factor 8 --reuse-index --tag exp6_zrg

run exp3 experiments/exp3_scalability.py --synthetic --sizes 5000 10000 20000 --queries 500

run exp2 experiments/exp2_ablation.py --synthetic --n 20000 --queries 300 --k 10 \
    --ef 80 --m 16 --efc 100

run figures experiments/make_figures.py

echo "===== ALL DONE"
cat "$SUM"
