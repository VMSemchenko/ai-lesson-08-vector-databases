"""
Generate embeddings for corpus and queries using a local SentenceTransformer model.

Usage:
  python src/embed.py                                   # defaults
  python src/embed.py --model BAAI/bge-small-en-v1.5    # explicit model
  python src/embed.py --batch-size 512                   # faster on GPU

Outputs:
  data/corpus_embeddings.npy   — (N, dim) float32
  data/corpus_ids.json         — list of N doc IDs (parallel to rows)
  data/query_embeddings.npy    — (Q, dim) float32
  data/query_ids.json          — list of Q query IDs
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _load_jsonl(path: Path):
    """Read JSONL returning (ids, texts)."""
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(str(obj["_id"]))
            texts.append(obj["text"])
    return ids, texts


def _embed_and_save(
    model: SentenceTransformer,
    ids: list,
    texts: list,
    out_npy: Path,
    out_ids: Path,
    batch_size: int,
    normalize: bool = True,
):
    if out_npy.exists() and out_ids.exists():
        print(f"   ⏭  Cache hit — {out_npy.name} already exists, skipping.")
        return

    print(f"   ⏳ Encoding {len(texts):,} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    np.save(out_npy, embeddings)
    with open(out_ids, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"   ✅ {out_npy.name}: shape {embeddings.shape}, {out_npy.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for BeIR/quora")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Encoding batch size")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"🔧  Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    # --- Corpus embeddings ---
    print("\n📄  Corpus embeddings:")
    corpus_ids, corpus_texts = _load_jsonl(data_dir / "corpus.jsonl")
    _embed_and_save(
        model, corpus_ids, corpus_texts,
        out_npy=data_dir / "corpus_embeddings.npy",
        out_ids=data_dir / "corpus_ids.json",
        batch_size=args.batch_size,
    )

    # --- Query embeddings ---
    print("\n🔍  Query embeddings:")
    query_ids, query_texts = _load_jsonl(data_dir / "queries.jsonl")
    _embed_and_save(
        model, query_ids, query_texts,
        out_npy=data_dir / "query_embeddings.npy",
        out_ids=data_dir / "query_ids.json",
        batch_size=args.batch_size,
    )

    print("\n🎉  All embeddings cached in:", data_dir)


if __name__ == "__main__":
    main()
