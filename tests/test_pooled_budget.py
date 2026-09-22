"""批次共享预算池（BCAS-P）的硬性质回归测试。

这些性质是报告里"硬预算恒成立"这句话的全部依据，必须由测试守住：
被测代码一旦改动导致性质失效，这里要立刻红，而不是等到报告里出现假结论。

  P1  批次总额：sum(cost_i) <= nq * B        （池只减不增，按实际消耗扣减）
  P2  单查询上限：cost_i <= cap_max          （允许下降段固定开销造成的极小超出，需报出不掩盖）
  P3  保底：每个查询至少拿到 min(cap_max, B) 的可用量
  P4  口径：池化评测不接受 warmup（预热会消耗共享池，破坏预算口径）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.adaptive import PooledBudgetSearcher, pooled_batch_eval  # noqa: E402
from src.data import make_synthetic  # noqa: E402
from src.hnsw import HNSW  # noqa: E402


@pytest.fixture(scope="module")
def fixture():
    ds = make_synthetic(n=800, dim=32, nq=60, k=10)
    idx = HNSW(dim=ds.dim, m=8, ef_construction=40, seed=0)
    idx.add_items(ds.train, verbose_every=0)
    return ds, idx


@pytest.mark.parametrize("budget", [40, 120, 400])
def test_batch_total_never_exceeds_pool(fixture, budget):
    """P1：批次总额硬上界。透支额必须为 0。"""
    ds, idx = fixture
    s = PooledBudgetSearcher(idx, k=10, budget=budget, cap_factor=8.0, gain_tau=0.05)
    st = pooled_batch_eval(s, ds.queries, ds.neighbors, 10)

    nq = int(ds.queries.shape[0])
    total = int(np.asarray(st["dists"]).sum())
    assert total <= nq * budget, f"批次总额 {total} 超过池上限 {nq * budget}"
    assert st["overdraft"] == 0, f"池透支 {st['overdraft']}，批次硬上界被突破"
    # 池的收支必须自洽
    assert st["pool_total"] - st["pool_remaining"] == total


@pytest.mark.parametrize("cap_factor", [2.0, 8.0])
def test_per_query_cap_respected(fixture, cap_factor):
    """P2：单查询上限。只允许下降段固定开销造成的极小超出，且必须被计数报出。"""
    ds, idx = fixture
    s = PooledBudgetSearcher(idx, k=10, budget=150, cap_factor=cap_factor, gain_tau=0.05)
    st = pooled_batch_eval(s, ds.queries, ds.neighbors, 10)

    dists = np.asarray(st["dists"])
    over = int((dists > s.cap_max).sum())
    assert over == st["overshoot"], "超出计数与实际不符"
    # 下降段开销应当只有几十次；给出的上限是宽松的健全性检查，不是精确等式
    assert st["overshoot_max"] <= 200, f"下降段超出 {st['overshoot_max']} 次，超出预期量级"


def test_floor_is_reserved_for_every_query(fixture):
    """P3：保底额度。每个查询至少有 min(cap_max, B) 的可用量（池充足时必然成立）。"""
    ds, idx = fixture
    B = 100
    s = PooledBudgetSearcher(idx, k=10, budget=B, cap_factor=8.0, gain_tau=0.05)
    st = pooled_batch_eval(s, ds.queries, ds.neighbors, 10)

    caps = np.asarray(st["caps"])
    assert caps.min() >= min(s.cap_max, B), (
        f"存在查询的可用额度 {caps.min()} 低于保底 {min(s.cap_max, B)}"
    )
    assert st["starved"] == 0, f"有 {st['starved']} 个查询的可用额度低于保底"


def test_pooled_eval_rejects_warmup(fixture):
    """P4：池化评测不能做预热，否则会消耗共享池、破坏预算口径。"""
    ds, idx = fixture
    s = PooledBudgetSearcher(idx, k=10, budget=100, cap_factor=8.0, gain_tau=0.05)
    with pytest.raises(ValueError, match="warmup"):
        pooled_batch_eval(s, ds.queries, ds.neighbors, 10, warmup=3)


def test_pool_must_be_begun_before_search(fixture):
    """未开批次就直接检索必须报错，而不是静默按空池运行。"""
    ds, idx = fixture
    s = PooledBudgetSearcher(idx, k=10, budget=100, cap_factor=8.0, gain_tau=0.05)
    with pytest.raises(RuntimeError, match="begin_batch"):
        s.search(ds.queries[0])


def test_invalid_cap_factor_rejected(fixture):
    """cap_factor < 1 会让单查询上限低于保底额度，是自相矛盾的配置。"""
    ds, idx = fixture
    with pytest.raises(ValueError, match="cap_factor"):
        PooledBudgetSearcher(idx, k=10, budget=100, cap_factor=0.5)
