"""ChromaDB — embedded persistent mode (no server needed)."""
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import chromadb
import numpy as np

from .base import VectorDB

_COLLECTION = "benchmark"
_BATCH_SIZE = 5000  # Chroma batch limit is ~5461 by default


class ChromaDB(VectorDB):
    """
    ChromaDB in persistent embedded mode.
    Uses cosine similarity via HNSW (Chroma's default).
    """

    def __init__(self, persist_dir: str = None):
        self._persist_dir = persist_dir or tempfile.mkdtemp(prefix="chroma_bench_")
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = None

    # ------------------------------------------------------------------ #
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        # Drop if exists
        try:
            self._client.delete_collection(_COLLECTION)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        n = len(ids)
        for start in range(0, n, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, n)
            self._collection.add(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
            )

    # ------------------------------------------------------------------ #
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
        )
        # results = {ids: [[...]], distances: [[...]]}
        out = []
        for doc_id, dist in zip(results["ids"][0], results["distances"][0]):
            # Chroma returns distance; for cosine, score = 1 - distance
            out.append((doc_id, 1.0 - dist))
        return out

    # ------------------------------------------------------------------ #
    def disk_size_mb(self) -> float:
        total = 0
        for p in Path(self._persist_dir).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024 * 1024)

    def cleanup(self) -> None:
        try:
            self._client.delete_collection(_COLLECTION)
        except Exception:
            pass
        shutil.rmtree(self._persist_dir, ignore_errors=True)
