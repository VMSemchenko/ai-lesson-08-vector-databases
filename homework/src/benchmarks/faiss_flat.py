"""FAISS Flat (brute-force) — 100 % recall baseline."""
import tempfile
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .base import VectorDB


class FaissFlat(VectorDB):
    """
    Flat inner-product index on L2-normalised vectors → cosine similarity.
    Guarantees exact (100 %) recall.
    """

    def __init__(self):
        self._index: faiss.IndexFlatIP | None = None
        self._ids: List[str] = []
        self._tmp_dir = tempfile.mkdtemp(prefix="faiss_flat_")

    # ------------------------------------------------------------------ #
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)  # type: ignore[arg-type]
        self._ids = list(ids)

        # persist so we can measure disk size
        self._index_path = Path(self._tmp_dir) / "flat.index"
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
        p = Path(self._tmp_dir) / "flat.index"
        return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
