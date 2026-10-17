"""数据层：合成数据（秒级冒烟）与公开数据集（正式实验）。

设计原则
--------
1. 合成数据自带 ground truth，冷启动零下载即可跑通全链路；
2. SIFT1M 走 ann-benchmarks 公开 HDF5，自带精确 top-k，不需要自己算真值；
3. 对底库做子采样后 **必须重算 ground truth** —— 原真值过滤一遍不等于子集上的真值。
   这是很容易写错的一处，本项目单独提供 with_train_subset()。
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ANN_BENCH_URLS = {
    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "fashion-mnist-784-euclidean": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
}


@dataclass
class Dataset:
    """近邻检索数据集。所有距离均为**平方欧氏距离**（省掉 sqrt，排序不变）。"""

    name: str
    train: np.ndarray      # (n,  d)  float32  底库
    queries: np.ndarray    # (nq, d)  float32  查询
    neighbors: np.ndarray  # (nq, k)  int32    精确 top-k 的 id
    distances: np.ndarray  # (nq, k)  float32  对应平方距离

    @property
    def n(self) -> int:
        return int(self.train.shape[0])

    @property
    def nq(self) -> int:
        return int(self.queries.shape[0])

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])

    @property
    def k(self) -> int:
        return int(self.neighbors.shape[1])

    def describe(self) -> str:
        return (
            f"{self.name}: n={self.n:,} nq={self.nq:,} dim={self.dim} k={self.k} "
            f"mem={self.train.nbytes / 2**20:.1f}MB"
        )

    def with_train_subset(
        self, n: int, seed: int = 0, recompute_truth: bool = True
    ) -> "Dataset":
        """取底库的一个随机子集，构造新数据集。

        recompute_truth=True 时用暴力搜索重算真值（正确但慢）；
        设 False 只用于快速实验，**报告中不得使用**，会低估召回率。
        """
        if n >= self.n:
            return self
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(self.n, size=n, replace=False))
        train = np.ascontiguousarray(self.train[sel])
        if recompute_truth:
            nb, dist = brute_force_knn(train, self.queries, self.k)
        else:  # 仅调试用：把落在子集内的原真值保留，其余丢弃
            remap = -np.ones(self.n, dtype=np.int64)
            remap[sel] = np.arange(n)
            nb = np.full(self.neighbors.shape, -1, dtype=np.int32)
            for i in range(self.nq):
                mapped = remap[self.neighbors[i]]
                mapped = mapped[mapped >= 0]
                nb[i, : mapped.size] = mapped
            dist = np.full(nb.shape, np.nan, dtype=np.float32)
        return Dataset(f"{self.name}[n={n}]", train, self.queries, nb, dist)


def brute_force_knn(
    train: np.ndarray, queries: np.ndarray, k: int, batch: Optional[int] = None,
    mem_budget_mb: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """精确 top-k（平方欧氏）。返回 (ids, 平方距离)。

    批大小按内存预算自动决定：单批的 (batch, n) float32 距离矩阵不超过 mem_budget_mb。
    否则在大底库（如 SIFT1M 的 100 万点）上会一次性申请 GB 级内存。
    """
    train = np.ascontiguousarray(train, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n = int(train.shape[0])
    if batch is None:
        batch = max(1, int(mem_budget_mb * 2**20 / max(n * 4, 1)))
    tn = np.einsum("ij,ij->i", train, train)
    ids = np.empty((queries.shape[0], k), dtype=np.int32)
    dst = np.empty((queries.shape[0], k), dtype=np.float32)
    for s in range(0, queries.shape[0], batch):
        e = min(s + batch, queries.shape[0])
        q = queries[s:e]
        d2 = tn[None, :] + np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ train.T)
        np.maximum(d2, 0.0, out=d2)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dv = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(dv, axis=1)
        ids[s:e] = np.take_along_axis(part, order, axis=1)
        dst[s:e] = np.take_along_axis(dv, order, axis=1)
    return ids, dst


def make_synthetic(
    n: int = 20000,
    dim: int = 64,
    nq: int = 200,
    k: int = 10,
    n_clusters: int = 32,
    spread: float = 0.25,
    seed: int = 0,
) -> Dataset:
    """带聚类结构的高斯合成数据。

    为什么不生成均匀随机点：均匀 i.i.d. 高维点的近邻结构退化（距离集中现象），
    图索引与暴力搜索的表现差异被抹平，实验失去鉴别力。真实向量（SIFT / GloVe /
    图像 embedding）都有簇结构，所以合成数据也必须带簇。
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(n_clusters, dim)).astype(np.float32)
    a_tr = rng.integers(0, n_clusters, size=n)
    a_q = rng.integers(0, n_clusters, size=nq)
    train = np.ascontiguousarray(
        centers[a_tr] + rng.normal(0.0, spread, size=(n, dim)), dtype=np.float32
    )
    queries = np.ascontiguousarray(
        centers[a_q] + rng.normal(0.0, spread, size=(nq, dim)), dtype=np.float32
    )
    nb, dist = brute_force_knn(train, queries, k)
    # nq 必须进 name：它让随机流偏移进而改变 train 本身。两个 nq 若共用同一个 name，
    # 索引缓存会误判为同一份底库而静默复用错误的图（见 _common.data_fingerprint）。
    return Dataset(f"synthetic(n={n},dim={dim},clusters={n_clusters},nq={nq})",
                   train, queries, nb, dist)


def download(url: str, dest: str, progress: Optional[Callable[[int, int], None]] = None) -> str:
    """带断点续传的下载。

    两个坑（都踩过，写在这里省得复现者再踩）：
    1. ann-benchmarks.com 的 CDN 对**没有 User-Agent** 的请求直接返回 403。
       用 requests/urllib 默认头会失败，必须显式带 UA。
    2. SIFT1M 有 525MB，中途断线重来一次很痛苦。`.part` 文件存在时用
       `Range: bytes=<已下载字节数>-` 续传；服务端不支持时退回整文件重下。
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cacc-vecsearch/1.0)"}

    got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if got:
        headers["Range"] = f"bytes={got}-"
        print(f"检测到未完成文件，尝试从 {got / 2**20:.1f}MB 续传")

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (416, 501):                  # 服务端不支持 Range / 已下完
            os.remove(tmp)
            got = 0
            resp = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": headers["User-Agent"]}), timeout=60)
        else:
            raise

    resuming = resp.status == 206                    # 206 = Partial Content
    if not resuming and got:
        got = 0                                      # 服务端忽略了 Range，整文件重下
    total = int(resp.headers.get("Content-Length") or 0) + got

    mode = "ab" if resuming else "wb"
    with resp, open(tmp, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if progress:
                progress(got, total)
    os.replace(tmp, dest)
    return dest


def load_hdf5(name: str, auto_download: bool = True, path: Optional[str] = None) -> Dataset:
    """读取 ann-benchmarks 格式的 HDF5。"""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "缺少 h5py。执行：pip install h5py\n（或用 make_synthetic() 先跑通全链路）"
        ) from exc

    path = path or os.path.join(DATA_DIR, f"{name}.hdf5")
    if not os.path.exists(path):
        if not auto_download:
            raise FileNotFoundError(path)
        if name not in ANN_BENCH_URLS:
            raise KeyError(f"未知数据集 {name}，可选：{list(ANN_BENCH_URLS)}")
        url = ANN_BENCH_URLS[name]
        print(f"下载 {name} -> {path}\n来源 {url}")

        def _p(got: int, total: int) -> None:
            if total:
                print(f"\r  {got / 2**20:7.1f} / {total / 2**20:7.1f} MB", end="", flush=True)

        download(url, path, _p)
        print()

    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        queries = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
        distances = np.asarray(f["distances"], dtype=np.float32)
    return Dataset(name, train, queries, neighbors, distances)
