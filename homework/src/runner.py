"""
Benchmark runner — runs all 5 vector DBs and outputs results.csv.

Usage:
  python src/runner.py
  python src/runner.py --output results/results.csv
  python src/runner.py --dbs faiss_flat faiss_hnsw   # run only subset
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

# Ensure src/ is on the path so `benchmarks` sub-package is importable
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
WARMUP_QUERIES = 50   # first N queries NOT measured (cold cache, JIT)
NUM_REPEATS = 3       # repeat measurement, take median


def _recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Recall@K = |retrieved ∩ relevant| / min(K, |relevant|)."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant))


def _mrr_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """MRR@K = 1 / rank of first correct result (0 if none)."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------
def benchmark_db(
    db,
    name: str,
    doc_vectors: np.ndarray,
    doc_ids: List[str],
    query_vectors: np.ndarray,
    query_ids: List[str],
    qrels: Dict[str, set],
    top_k: int = 10,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  🔬 Benchmarking: {name}")
    print(f"{'='*60}")

    # === INDEX ===
    print(f"  📦 Indexing {len(doc_ids):,} vectors...")
    t0 = time.perf_counter()
    db.index(doc_vectors, ids=doc_ids)
    index_time = time.perf_counter() - t0
    print(f"     ✅ Done in {index_time:.1f}s")

    # === WARMUP ===
    print(f"  🔥 Warming up ({WARMUP_QUERIES} queries)...")
    for q_vec in query_vectors[:WARMUP_QUERIES]:
        db.search(q_vec, top_k=top_k)

    # === MEASURED QUERIES ===
    n_queries = len(query_vectors)
    print(f"  📊 Measuring {n_queries:,} queries × {NUM_REPEATS} repeats...")

    all_latencies: List[List[float]] = []
    recalls: List[float] = []
    mrrs: List[float] = []

    for repeat in range(NUM_REPEATS):
        latencies = []
        desc = f"  repeat {repeat + 1}/{NUM_REPEATS}"
        for q_vec, q_id in tqdm(
            zip(query_vectors, query_ids), total=n_queries, desc=desc, leave=False
        ):
            t0 = time.perf_counter()
            results = db.search(q_vec, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            if repeat == 0:
                retrieved_ids = [doc_id for doc_id, _score in results]
                relevant = qrels.get(q_id, set())
                recalls.append(_recall_at_k(retrieved_ids, relevant, top_k))
                mrrs.append(_mrr_at_k(retrieved_ids, relevant, top_k))
        all_latencies.append(latencies)

    # median across repeats per query, then percentiles
    latencies_arr = np.median(np.array(all_latencies), axis=0)

    result = {
        "db_name": name,
        "index_time_sec": round(index_time, 2),
        "disk_mb": round(db.disk_size_mb(), 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "mrr_at_10": round(float(np.mean(mrrs)), 4),
        "num_queries": n_queries,
    }

    print(f"\n  📋 Results for {name}:")
    for k, v in result.items():
        if k != "db_name":
            print(f"     {k:>20s}: {v}")

    return result


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_qrels(path: Path) -> Dict[str, Set[str]]:
    """Load qrels.tsv → {query_id: {relevant_doc_ids}}."""
    qrels: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            q_id, c_id, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(q_id, set()).add(c_id)
    return qrels


# ---------------------------------------------------------------------------
# DB factory
# ---------------------------------------------------------------------------
def _get_db_instances() -> Dict[str, object]:
    """Return name → VectorDB instance for all 5 required DBs."""
    from benchmarks.faiss_flat import FaissFlat
    from benchmarks.faiss_hnsw import FaissHNSW
    from benchmarks.qdrant_db import QdrantDB
    from benchmarks.chroma_db import ChromaDB
    from benchmarks.pgvector_db import PgvectorDB

    return {
        "FAISS_Flat": FaissFlat(),
        "FAISS_HNSW": FaissHNSW(M=32, ef_search=128, ef_construction=200),
        "Qdrant": QdrantDB(),
        "Chroma": ChromaDB(),
        "pgvector": PgvectorDB(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vector DB Benchmark Runner")
    parser.add_argument("--output", default="results/results.csv", help="Output CSV path")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--dbs", nargs="*", default=None, help="Subset of DBs to run")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("📥  Loading embeddings & qrels...")
    doc_vectors = np.load(data_dir / "corpus_embeddings.npy")
    query_vectors = np.load(data_dir / "query_embeddings.npy")
    with open(data_dir / "corpus_ids.json") as f:
        doc_ids = json.load(f)
    with open(data_dir / "query_ids.json") as f:
        query_ids = json.load(f)
    qrels = load_qrels(data_dir / "qrels.tsv")

    print(f"   docs:    {doc_vectors.shape}")
    print(f"   queries: {query_vectors.shape}")
    print(f"   qrels:   {len(qrels):,} queries with relevance judgments")

    # Filter queries to those with qrels
    mask = [i for i, qid in enumerate(query_ids) if qid in qrels]
    query_vectors = query_vectors[mask]
    query_ids = [query_ids[i] for i in mask]
    print(f"   queries with qrels: {len(query_ids):,}")

    # Get DB instances
    all_dbs = _get_db_instances()
    if args.dbs:
        all_dbs = {k: v for k, v in all_dbs.items() if k in args.dbs}

    # Run benchmarks
    results = []
    for name, db in all_dbs.items():
        try:
            result = benchmark_db(
                db, name,
                doc_vectors, doc_ids,
                query_vectors, query_ids,
                qrels, top_k=args.top_k,
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
        finally:
            db.cleanup()

    # Save CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾  Results saved to: {output_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("  📋 SUMMARY")
    print(f"{'='*80}")
    header = f"{'DB':<15} {'Recall@10':>10} {'MRR@10':>10} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'Index s':>10} {'Disk MB':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['db_name']:<15} "
            f"{r['recall_at_10']:>10.4f} "
            f"{r['mrr_at_10']:>10.4f} "
            f"{r['latency_p50_ms']:>10.3f} "
            f"{r['latency_p95_ms']:>10.3f} "
            f"{r['latency_p99_ms']:>10.3f} "
            f"{r['index_time_sec']:>10.2f} "
            f"{r['disk_mb']:>10.1f}"
        )


if __name__ == "__main__":
    main()
