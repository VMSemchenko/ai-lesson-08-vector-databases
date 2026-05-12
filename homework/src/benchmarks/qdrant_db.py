"""Qdrant — client/server via Docker (REST on port 6333)."""
import os
import uuid
from typing import List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .base import VectorDB

_COLLECTION = "benchmark"
_BATCH_SIZE = 1000  # upload batch size


class QdrantDB(VectorDB):
    """
    Qdrant running in Docker (docker-compose.yml → hw_qdrant).
    Uses cosine distance. HNSW params left at defaults.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection: str = _COLLECTION,
    ):
        self._host = host or os.getenv("QDRANT_HOST", "localhost")
        self._port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self._collection = collection
        self._client = QdrantClient(host=self._host, port=self._port, timeout=300)
        self._ids: List[str] = []

    # ------------------------------------------------------------------ #
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        dim = vectors.shape[1]
        self._ids = list(ids)

        # recreate collection
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # upload in batches
        n = len(ids)
        for start in range(0, n, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, n)
            points = [
                PointStruct(
                    id=start + i,
                    vector=vectors[start + i].tolist(),
                    payload={"doc_id": ids[start + i]},
                )
                for i in range(end - start)
            ]
            self._client.upsert(collection_name=self._collection, points=points)

        # build a reverse map: qdrant int id → doc_id
        self._id_map = {i: doc_id for i, doc_id in enumerate(ids)}

    # ------------------------------------------------------------------ #
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [(pt.payload["doc_id"], pt.score) for pt in result.points]

    # ------------------------------------------------------------------ #
    def disk_size_mb(self) -> float:
        info = self._client.get_collection(self._collection)
        # Qdrant reports in bytes via collection_info
        try:
            seg_info = info.points_count  # rough proxy
            # Use the disk data from collection info
            disk_bytes = 0
            for seg in (info.segments or []):
                disk_bytes += seg.get("disk_usage_bytes", 0)
            if disk_bytes > 0:
                return disk_bytes / (1024 * 1024)
        except Exception:
            pass
        # Fallback: estimate from vector count × dim × 4 bytes
        dim = info.config.params.vectors.size
        n = info.points_count or 0
        return (n * dim * 4) / (1024 * 1024)

    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        try:
            self._client.delete_collection(self._collection)
        except Exception:
            pass
