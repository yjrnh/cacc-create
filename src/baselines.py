"""基线索引。

分层设计，缺依赖不影响主链路：
  BruteForceIndex  —— 纯 numpy，永远可用。召回上界(1.0)、距离次数上界(n/query)
  FaissHNSW        —— faiss-cpu，工业界图索引基准
  FaissIVFPQ       —— faiss-cpu，量化路线对照（内存换召回）
  HnswlibIndex     —— hnswlib，HNSW 官方 C++ 实现

诚实说明（必须写进报告）：
  Faiss 与 hnswlib 都不对外暴露距离计算次数，只能测 QPS。这不是回避，而是本项目
  自实现索引的理由之一 —— 只有自实现才能拿到与硬件无关的算法层指标。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .counter import DistanceCounter


class BaseIndex:
    name = "base"
    counts_distances = True

    def add_items(self, vectors: np.ndarray) -> "BaseIndex":  # pragma: no cover
        raise NotImplementedError

    def search(self, q: np.ndarray, k: int, ef: Optional[int] = None):
        raise NotImplementedError

    def build_time_note(self) -> str:
        return ""


class BruteForceIndex(BaseIndex):
    name = "BruteForce"

    def __init__(self) -> None:
        self.vectors: Optional[np.ndarray] = None
        self.counter = DistanceCounter()

    def add_items(self, vectors: np.ndarray) -> "BruteForceIndex":
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        return self

    def search(self, q: np.ndarray, k: int, ef: Optional[int] = None):
        """分块精确搜索，内存占用与底库规模无关（约 64MB/块）。"""
        n, d = self.vectors.shape
        k = min(int(k), n)
        chunk = max(1, int(64 * 2**20 / (d * 4)))
        cd, ci = [], []
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            V = self.vectors[s:e]
            diff = V - q
            d2 = np.einsum("ij,ij->i", diff, diff)
            self.counter.add(e - s)
            kk = min(k, d2.shape[0])
            part = np.argpartition(d2, kk - 1)[:kk]
            cd.append(d2[part])
            ci.append(part.astype(np.int32) + s)
        D = np.concatenate(cd)
        I = np.concatenate(ci)
        order = np.argsort(D)[:k]
        return I[order].astype(np.int32), D[order].astype(np.float32)


class FaissHNSW(BaseIndex):
    name = "faiss-HNSW"
    counts_distances = False

    def __init__(self, m: int = 16, ef_construction: int = 100, ef_search: int = 64) -> None:
        self.m, self.ef_construction, self.ef_search = int(m), int(ef_construction), int(ef_search)
        self.index = None

    @staticmethod
    def available() -> bool:
        try:
            import faiss  # noqa: F401
        except Exception:
            return False
        return True

    def add_items(self, vectors: np.ndarray) -> "FaissHNSW":
        import faiss

        v = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexHNSWFlat(int(v.shape[1]), self.m, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.add(v)
        return self

    def search(self, q: np.ndarray, k: int, ef: Optional[int] = None):
        self.index.hnsw.efSearch = int(ef or self.ef_search)
        d, i = self.index.search(np.ascontiguousarray(q[None, :], dtype=np.float32), int(k))
        return i[0].astype(np.int32), d[0].astype(np.float32)


class FaissIVFPQ(BaseIndex):
    name = "faiss-IVFPQ"
    counts_distances = False

    def __init__(self, nlist: int = 256, m_sub: int = 16, nbits: int = 8) -> None:
        self.nlist, self.m_sub, self.nbits = int(nlist), int(m_sub), int(nbits)
        self.index = None

    @staticmethod
    def available() -> bool:
        return FaissHNSW.available()

    def add_items(self, vectors: np.ndarray) -> "FaissIVFPQ":
        import faiss

        v = np.ascontiguousarray(vectors, dtype=np.float32)
        d = int(v.shape[1])
        quant = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quant, d, self.nlist, self.m_sub, self.nbits)
        self.index.train(v)
        self.index.add(v)
        return self

    def search(self, q: np.ndarray, k: int, ef: Optional[int] = None):
        if ef is not None:
            self.index.nprobe = int(ef)
        d, i = self.index.search(np.ascontiguousarray(q[None, :], dtype=np.float32), int(k))
        return i[0].astype(np.int32), d[0].astype(np.float32)


class HnswlibIndex(BaseIndex):
    name = "hnswlib"
    counts_distances = False

    def __init__(self, m: int = 16, ef_construction: int = 100, ef_search: int = 64) -> None:
        self.m, self.ef_construction, self.ef_search = int(m), int(ef_construction), int(ef_search)
        self.index = None

    @staticmethod
    def available() -> bool:
        try:
            import hnswlib  # noqa: F401
        except Exception:
            return False
        return True

    def add_items(self, vectors: np.ndarray) -> "HnswlibIndex":
        import hnswlib

        v = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = hnswlib.Index(space="l2", dim=int(v.shape[1]))
        self.index.init_index(max_elements=int(v.shape[0]), ef_construction=self.ef_construction, M=self.m)
        self.index.add_items(v, np.arange(v.shape[0], dtype=np.int64))
        return self

    def search(self, q: np.ndarray, k: int, ef: Optional[int] = None):
        self.index.set_ef(int(ef or self.ef_search))
        i, d = self.index.knn_query(np.ascontiguousarray(q[None, :], dtype=np.float32), k=int(k))
        return i[0].astype(np.int32), d[0].astype(np.float32)


def make_baselines(dim: int, m: int = 16, ef_construction: int = 100) -> Dict[str, BaseIndex]:
    """按可用性装配基线集合，缺失的只提示不报错。"""
    out: Dict[str, BaseIndex] = {"BruteForce": BruteForceIndex()}
    if FaissHNSW.available():
        out["faiss-HNSW"] = FaissHNSW(m=m, ef_construction=ef_construction)
        out["faiss-IVFPQ"] = FaissIVFPQ()
    else:
        print("[跳过] 未安装 faiss-cpu，基线集合中不含 faiss")
    if HnswlibIndex.available():
        out["hnswlib"] = HnswlibIndex(m=m, ef_construction=ef_construction)
    else:
        print("[跳过] 未安装 hnswlib，基线集合中不含 hnswlib")
    return out
