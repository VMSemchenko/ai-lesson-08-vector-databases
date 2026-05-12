"""
Load the BeIR/quora dataset and save corpus, queries, and qrels to data/.

Output files:
  data/corpus.jsonl   — ~523K documents  {_id, text}
  data/queries.jsonl  — ~10K queries     {_id, text}
  data/qrels.tsv      — relevance judgments (query_id, corpus_id, score)
"""
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("📥  Downloading BeIR/quora from HuggingFace...")
    ds = load_dataset("BeIR/quora", "corpus")
    ds_queries = load_dataset("BeIR/quora", "queries")

    # --- Corpus ---
    corpus_path = DATA_DIR / "corpus.jsonl"
    print(f"💾  Saving corpus → {corpus_path}")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds["corpus"], desc="corpus"):
            doc = {
                "_id": row["_id"],
                "text": (row.get("title", "") + " " + row.get("text", "")).strip(),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"   ✅ {sum(1 for _ in open(corpus_path)):,} documents")

    # --- Queries ---
    queries_path = DATA_DIR / "queries.jsonl"
    print(f"💾  Saving queries → {queries_path}")
    with open(queries_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds_queries["queries"], desc="queries"):
            doc = {"_id": row["_id"], "text": row["text"].strip()}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"   ✅ {sum(1 for _ in open(queries_path)):,} queries")

    # --- Qrels (relevance judgments) ---
    print("📥  Downloading qrels split...")
    # BeIR/quora-qrels has a default split with columns: query-id, corpus-id, score
    try:
        ds_qrels = load_dataset("BeIR/quora-qrels")
        qrels_split = ds_qrels.get("test") or ds_qrels.get("validation") or ds_qrels[list(ds_qrels.keys())[0]]
    except Exception:
        # Fallback: some versions of the dataset have qrels embedded differently
        ds_qrels = load_dataset("BeIR/quora", "default")
        qrels_split = ds_qrels["test"] if "test" in ds_qrels else ds_qrels[list(ds_qrels.keys())[0]]

    qrels_path = DATA_DIR / "qrels.tsv"
    print(f"💾  Saving qrels → {qrels_path}")
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("query_id\tcorpus_id\tscore\n")
        for row in tqdm(qrels_split, desc="qrels"):
            qid = row.get("query-id") or row.get("query_id")
            cid = row.get("corpus-id") or row.get("corpus_id")
            score = row.get("score", 1)
            f.write(f"{qid}\t{cid}\t{score}\n")
    n_qrels = sum(1 for _ in open(qrels_path)) - 1  # minus header
    print(f"   ✅ {n_qrels:,} relevance judgments")

    print("\n🎉  Done! Data saved to:", DATA_DIR)


if __name__ == "__main__":
    main()
