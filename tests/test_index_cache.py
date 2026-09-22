"""索引缓存的回归测试。

缓存是这套实验里最危险的组件：键拼错了不会抛异常，只会安静地复用一张错误的图，
然后所有下游数字全部作废，而你根本看不出来。所以这里把四条不变量钉死：

  1. 首次构建返回真实耗时，并落下 sidecar 元数据；
  2. 二次调用命中缓存，返回的是**上次存档的真实耗时**，而不是 0 / None；
  3. 没有 sidecar 的历史缓存命中时返回 None（调用方渲染成 "-"），不谎报为 0；
  4. 决定图结构却不在键里的参数（seed / early_stop_eps / level_mult）直接报错，
     存在结构不一致的缓存文件时校验报错，而不是复用。

跑法：  python -m pytest tests -q      （没有 pytest 也可以直接 python tests/test_index_cache.py）
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import _common  # noqa: E402
from experiments._common import build_or_load_index, _meta_path  # noqa: E402
from src.data import make_synthetic  # noqa: E402


def _tiny():
    ds = make_synthetic(n=400, dim=16, nq=10, k=5)
    return ds


def test_build_then_hit_reports_archived_time():
    ds = _tiny()
    with tempfile.TemporaryDirectory() as tmp:
        old = _common.CACHE
        _common.CACHE = tmp
        try:
            idx1, bt1 = build_or_load_index(ds.train, ds.dim, ds.name, m=8, efc=40)
            assert isinstance(bt1, float) and bt1 > 0, f"首次构建应返回正耗时，得到 {bt1!r}"
            pk = [f for f in os.listdir(tmp) if f.endswith(".pkl")]
            assert len(pk) == 1, f"应只有一个 .pkl，实际 {os.listdir(tmp)}"
            assert os.path.exists(_meta_path(os.path.join(tmp, pk[0]))), "sidecar 元数据未落盘"

            idx2, bt2 = build_or_load_index(ds.train, ds.dim, ds.name, m=8, efc=40)
            assert bt2 is not None, "有 sidecar 时命中缓存必须报出存档耗时，不能是 None"
            assert abs(bt2 - bt1) < 1e-6, f"存档耗时 {bt2} 与首次构建 {bt1} 不一致"
            assert len(os.listdir(tmp)) == 2, "命中缓存不应产生新的 .pkl（应只有 pkl + meta）"

            # 复用的图必须真的可用，且构图距离计数随图一起被恢复
            assert idx1.counter.count == idx2.counter.count > 0
        finally:
            _common.CACHE = old


def test_hit_without_sidecar_reports_none_not_zero():
    ds = _tiny()
    with tempfile.TemporaryDirectory() as tmp:
        old = _common.CACHE
        _common.CACHE = tmp
        try:
            _, bt1 = build_or_load_index(ds.train, ds.dim, ds.name, m=8, efc=40)
            assert bt1 and bt1 > 0
            # 模拟历史缓存：只留 pkl，删掉 sidecar
            pkl = [f for f in os.listdir(tmp) if f.endswith(".pkl")][0]
            os.remove(_meta_path(os.path.join(tmp, pkl)))

            _, bt2 = build_or_load_index(ds.train, ds.dim, ds.name, m=8, efc=40)
            assert bt2 is None, "无元数据时必须返回 None，绝不能谎报 0（会被读成'建索引不花钱'）"
        finally:
            _common.CACHE = old


def test_structural_params_outside_key_are_rejected():
    ds = _tiny()
    with tempfile.TemporaryDirectory() as tmp:
        old = _common.CACHE
        _common.CACHE = tmp
        try:
            for kwargs in ({"seed": 7}, {"early_stop_eps": 0.01}, {"level_mult": 0.7}):
                try:
                    build_or_load_index(ds.train, ds.dim, ds.name, m=8, efc=40, **kwargs)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{kwargs} 改变图结构却不在缓存键里，应当报错而不是复用")
        finally:
            _common.CACHE = old


def test_mismatched_cache_file_is_caught():
    """键碰撞 / 手放错文件：必须报错，不能拿错图跑出漂亮数字。"""
    ds = _tiny()
    with tempfile.TemporaryDirectory() as tmp:
        old = _common.CACHE
        _common.CACHE = tmp
        try:
            _, bt = build_or_load_index(ds.train, ds.dim, ds.name, m=8, efc=40)
            assert bt and bt > 0
            # 把正确名字的缓存文件名改成另一个参数组合的键，模拟键拼错
            from experiments._common import index_key
            good = [f for f in os.listdir(tmp) if f.endswith(".pkl")][0]
            bad_key = index_key(ds.name, ds.train.shape[0], 32, 200)
            os.rename(os.path.join(tmp, good), os.path.join(tmp, bad_key + ".pkl"))

            try:
                build_or_load_index(ds.train, ds.dim, ds.name, m=32, efc=200)
            except RuntimeError as e:
                assert "不符" in str(e)
            else:
                raise AssertionError("缓存内容与请求参数不符时必须报错")
        finally:
            _common.CACHE = old


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n全部通过（{len(fns)} 项）")
