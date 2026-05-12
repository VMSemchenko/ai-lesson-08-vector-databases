"""FAISS HNSW — approximate search with tuneable M and efSearch."""
import tempfile
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .base import VectorDB


class FaissHNSW(VectorDB):
    """
    HNSW index via FAISS (inner-product on L2-normalised vectors ≈ cosine).
    Key knobs: M (graph connectivity), efSearch (query-time beam width).
    """

    def __init__(self, M: int = 32, ef_search: int = 128, ef_construction: int = 200):
        self._M = M
        self._ef_search = ef_search
        self._ef_construction = ef_construction
        self._index: faiss.IndexHNSWFlat | None = None
        self._ids: List[str] = []
        self._tmp_dir = tempfile.mkdtemp(prefix="faiss_hnsw_")

    # ------------------------------------------------------------------ #
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        dim = vectors.shape[1]
        self._index = faiss.IndexHNSWFlat(dim, self._M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self._ef_construction
        self._index.hnsw.efSearch = self._ef_search
        self._index.add(vectors)  # type: ignore[arg-type]
        self._ids = list(ids)

        # persist for disk size measurement
        self._index_path = Path(self._tmp_dir) / "hnsw.index"
        faiss.write_index(self._index, str(self._index_path))

    # ------------------------------------------------------------------ #
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        q = query_vec.reshape(1, -1)
        scores, indices = self._index.search(q, top_k)  # type: ignore[union-attr]
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._ids[idx], float(score)))
        return results

    # ------------------------------------------------------------------ #
    def disk_size_mb(self) -> float:
        p = Path(self._tmp_dir) / "hnsw.index"
        return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
