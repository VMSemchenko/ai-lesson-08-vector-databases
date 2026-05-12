"""pgvector — Postgres + HNSW index via Docker."""
import os
from typing import List, Tuple

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .base import VectorDB

_TABLE = "bench_vectors"
_BATCH_SIZE = 1000


class PgvectorDB(VectorDB):
    """
    pgvector running in Docker (docker-compose.yml → hw_pgvector).
    Uses HNSW index with cosine distance.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        dbname: str = None,
    ):
        self._host = host or os.getenv("POSTGRES_HOST", "localhost")
        self._port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self._user = user or os.getenv("POSTGRES_USER", "bench")
        self._password = password or os.getenv("POSTGRES_PASSWORD", "bench")
        self._dbname = dbname or os.getenv("POSTGRES_DB", "bench")
        self._conn = None
        self._dim = None

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                dbname=self._dbname,
                autocommit=True,
            )
            # Must create extension before registering the vector type
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    # ------------------------------------------------------------------ #
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        self._dim = vectors.shape[1]
        conn = self._connect()

        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {_TABLE}")
            cur.execute(f"""
                CREATE TABLE {_TABLE} (
                    doc_id TEXT PRIMARY KEY,
                    embedding vector({self._dim})
                )
            """)

            # Bulk insert with COPY for performance
            print(f"      pgvector: inserting {len(ids):,} vectors...")
            with cur.copy(f"COPY {_TABLE} (doc_id, embedding) FROM STDIN") as copy:
                for doc_id, vec in zip(ids, vectors):
                    vec_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
                    copy.write_row((doc_id, vec_str))

            # Build HNSW index (cosine)
            print("      pgvector: building HNSW index...")
            cur.execute(f"""
                CREATE INDEX ON {_TABLE}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)

            # Set ef_search for queries
            cur.execute("SET hnsw.ef_search = 128")

    # ------------------------------------------------------------------ #
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        conn = self._connect()
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT doc_id, 1 - (embedding <=> %s::vector) AS score
                FROM {_TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, vec_str, top_k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    # ------------------------------------------------------------------ #
    def disk_size_mb(self) -> float:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT pg_total_relation_size('{_TABLE}') +
                       COALESCE((SELECT SUM(pg_relation_size(indexrelid))
                                 FROM pg_index WHERE indrelid = '{_TABLE}'::regclass), 0)
            """)
            total_bytes = cur.fetchone()[0]
        return total_bytes / (1024 * 1024)

    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {_TABLE}")
        except Exception:
            pass
        finally:
            if self._conn and not self._conn.closed:
                self._conn.close()
