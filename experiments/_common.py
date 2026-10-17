"""实验公共工具：路径、输出、索引缓存。

索引缓存是硬需求不是优化：单次构图在 n=5 万上要十几分钟，而一个实验往往要
在同一张图上反复跑不同参数。缓存键必须包含**决定图结构的全部参数**
（n / m / ef_construction / 数据源 / 构图开关），否则会静默复用到错误的图 ——
那是最难查的一类错误。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "index_cache")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72, flush=True)


# 决定图结构的构图开关。它们必须全部进缓存键，不能"没传就不算"。
_STRUCT_FLAGS = ("neighbor_mode", "extend_candidates", "keep_pruned")


def index_key(name: str, n: int, m: int, efc: int, **flags) -> str:
    """构造缓存键。任何影响图结构的量都要进 key。

    缺省参数与显式 None 必须归一成同一个键。否则 `index_key(...)` 和
    `index_key(..., neighbor_mode=None)` 会为同一张图算出两个不同的键 —— 结果是
    同一张图被存两份（浪费），或者更糟：把某个默认值改掉之后，旧缓存被静默复用。
    """
    for k in _STRUCT_FLAGS:
        flags.setdefault(k, None)
    extra = "-".join(f"{k}{flags[k]}" for k in sorted(flags))
    raw = f"{name}|n{n}|m{m}|efc{efc}|{extra}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    safe = name.replace("|", "-").replace("/", "-").replace(" ", "")
    return f"{safe}_n{n}_m{m}_efc{efc}_{digest}"


def data_fingerprint(train) -> str:
    """底库内容指纹。

    为什么必须校验内容而不只校验 shape：合成数据的生成顺序是
    centers → 训练标签 → **查询标签(nq)** → 训练向量 → 查询向量，
    nq 变化会让随机流偏移，从而产出**不同的底库**。而数据集 name 里不含 nq，
    于是 nq=200 与 nq=300 会算出同一个缓存键 —— 后者静默复用前者的图，
    图上节点编号指向另一批向量，召回率会崩到接近 0（实测 0.0237 vs 0.9060），
    且全程不报任何错。这类错误只有靠内容指纹才能拦住。
    """
    arr = np.ascontiguousarray(train)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _meta_path(path: str) -> str:
    return (path[:-4] if path.endswith(".pkl") else path) + ".meta.json"


def build_or_load_index(train, dim: int, name: str, *, m: int, efc: int, seed: int = 0,
                        reuse: bool = True, verbose_every: int = 0, **hnsw_kwargs):
    """返回 (index, build_seconds)。

    build_seconds 的取值有三种，调用方必须区分：
      * float —— 本次真的建了图，或者是缓存命中但读到了 sidecar 里存档的真实耗时；
      * None  —— 缓存命中但**没有** sidecar（历史缓存 / 手工放进去的 pkl），耗时不可知。

    **一定不能把 None 当成 0**：报告里"构建耗时 0 秒"会被读成建索引不花钱，
    而实际上只是复用了上次的结果。渲染成 "-" 才是诚实做法。

    sidecar 元数据单独存文件而不是塞进 pickle，是为了让"图"和"关于图的元信息"
    可以独立读取 —— 想查上一次构图花了多久，不必反序列化整个索引。

    hnsw_kwargs 里放构图开关（neighbor_mode / extend_candidates / keep_pruned），
    它们会进缓存键。
    """
    from src.hnsw import HNSW

    # 这三个参数同样决定图结构，但不在 index_key 里。与其默默复用错的图，
    # 不如直接拒绝 —— 静默复用是这类缓存最难查的一类故障。
    if seed != 0:
        raise ValueError("seed 影响图结构但不在缓存键中；请把它并入 index_key，或清空 data/index_cache")
    if hnsw_kwargs.get("early_stop_eps") not in (None, 0.0):
        raise ValueError("非零 early_stop_eps 会让构图期搜索提前终止，图结构与默认值不同；"
                         "请把它并入 index_key 后再启用缓存")
    if hnsw_kwargs.get("level_mult") is not None:
        raise ValueError("自定义 level_mult 影响层高采样，图结构与默认值不同；请并入 index_key")
    # 这两个键只是为了上面的校验而检查。"显式传 None" 与 "不传" 等价，
    # 但 HNSW.__init__ 收到 early_stop_eps=None 会直接 float(None) 崩掉，
    # 所以在这里把 None 摘掉，调用方不必知道这个细节。
    for _k in ("early_stop_eps", "level_mult"):
        if hnsw_kwargs.get(_k) is None:
            hnsw_kwargs.pop(_k, None)

    key = index_key(name, train.shape[0], m, efc, **{k: hnsw_kwargs.get(k) for k in
                                                     ("neighbor_mode", "extend_candidates", "keep_pruned")})
    path = os.path.join(CACHE, key + ".pkl")
    meta = _meta_path(path)
    if reuse and os.path.exists(path):
        t0 = time.perf_counter()
        idx = HNSW.load(path)
        load_s = time.perf_counter() - t0
        # 键是字符串拼出来的，拼错了不会报错只会复用错图。这里用索引自身的属性回查一次。
        problems = []
        if int(idx.dim) != int(dim):
            problems.append(f"dim 缓存={idx.dim} 期望={dim}")
        if idx.vectors is None or int(idx.vectors.shape[0]) != int(train.shape[0]):
            problems.append(f"底库大小 缓存={None if idx.vectors is None else idx.vectors.shape[0]} "
                            f"期望={train.shape[0]}")
        if int(idx.m) != int(m) or int(idx.ef_construction) != int(efc):
            problems.append(f"m/efc 缓存={idx.m}/{idx.ef_construction} 期望={m}/{efc}")
        bt = None
        if os.path.exists(meta):
            try:
                with open(meta, "r", encoding="utf-8") as fh:
                    _m = json.load(fh)
                bt = _m.get("build_seconds")
                fp = _m.get("data_fp")
                if fp and fp != data_fingerprint(train):
                    problems.append(f"底库内容指纹 缓存={fp} 期望={data_fingerprint(train)}"
                                    "（同一 name/n 下数据已变化，通常是 nq 不同导致合成数据随机流偏移）")
                if not fp:
                    problems.append("缓存缺少底库指纹（历史缓存），无法确认图与底库匹配")
            except (OSError, ValueError):
                bt = None
        else:
            problems.append("缓存缺少元数据，无法确认图与底库匹配")
        if problems:
            raise RuntimeError(f"缓存 {os.path.basename(path)} 与本次请求不符：{'；'.join(problems)}。"
                               "缓存键已失效，请删除该 .pkl 后重跑。")
        shown = f"{bt:.1f}s（存档值）" if bt is not None else "未知（无元数据）"
        print(f"  [cache] 复用 {os.path.basename(path)}  载入 {load_s:.1f}s | 构图耗时 {shown}")
        return idx, bt

    idx = HNSW(dim=dim, m=m, ef_construction=efc, seed=seed, **hnsw_kwargs)
    t0 = time.perf_counter()
    idx.add_items(train, verbose_every=verbose_every)
    bt = time.perf_counter() - t0
    idx.save(path)
    # 构图在大 n 上要十几分钟。存完立刻回查一次：如果磁盘写失败而脚本又在后面
    # 崩了（例如基线段的异常），这十几分钟就凭空虚耗。宁可在这里立刻报错。
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"索引写入失败：{path} 不存在或为空。构图耗时 {bt:.1f}s 已丢失，请检查磁盘。")
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({
            "key": key, "dataset": name, "n": int(train.shape[0]), "dim": int(dim),
            "m": m, "ef_construction": efc, "seed": seed,
            "build_seconds": bt, "build_dists": int(idx.counter.count),
            "data_fp": data_fingerprint(train),
            **{k: hnsw_kwargs.get(k) for k in ("neighbor_mode", "extend_candidates", "keep_pruned")},
        }, fh, ensure_ascii=False, indent=2)
    print(f"  [cache] 已缓存 {os.path.basename(path)}  构图 {bt:.1f}s", flush=True)
    return idx, bt


def save_rows(tag: str, rows: list[dict]) -> str:
    """同时落 CSV（utf-8-sig，Excel 直接打开不乱码）与 JSON。"""
    if not rows:
        return ""
    keys = list(rows[0].keys())
    cpath = os.path.join(RESULTS, f"{tag}.csv")
    with open(cpath, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, f"{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    return cpath
