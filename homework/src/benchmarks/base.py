"""Abstract interface for all vector DB implementations."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class VectorDB(ABC):
    """Common interface for FAISS / Qdrant / Chroma / pgvector."""

    @abstractmethod
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build an index from vectors.
        vectors: shape (N, dim), float32, L2-normalised for cosine
        ids: string IDs parallel to vectors
        """

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find top-K nearest vectors.
        query_vec: shape (dim,) — 1-D
        Returns: [(doc_id, score), ...] of length top_k
        """

    @abstractmethod
    def disk_size_mb(self) -> float:
        """Size of the index on disk in MB (0 if purely in-memory)."""

    def cleanup(self) -> None:
        """Close connections, delete temp files. Default: no-op."""
        pass
