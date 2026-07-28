#!/usr/bin/env python3
"""RAG 검색: 한국어 자연어 질문 -> 데이터셋에서 실제로 존재하는 top-k 프로젝트 반환.
build_retrieval_index.py를 먼저 실행해서 임베딩을 만들어둬야 한다."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def _load():
    model_name = (DATA_DIR / "retrieval_model.txt").read_text(encoding="utf-8").strip()
    model = SentenceTransformer(model_name)
    embeddings = np.load(DATA_DIR / "retrieval_embeddings.npy")
    metadata = json.loads((DATA_DIR / "retrieval_metadata.json").read_text(encoding="utf-8"))
    return model, embeddings, metadata


def search(query: str, top_k: int = 5):
    model, embeddings, metadata = _load()
    query_emb = model.encode(["query: " + query], normalize_embeddings=True).astype("float32")[0]
    scores = embeddings @ query_emb
    top_idx = np.argsort(-scores)[:top_k]
    return [
        {
            "score": float(scores[i]),
            "title": metadata[i]["title"],
            "category": metadata[i]["category"],
            "difficulty": metadata[i]["difficulty"],
            "hardware": metadata[i]["hardware"],
            "description": metadata[i]["description"],
            "url": metadata[i]["url"],
        }
        for i in top_idx
    ]


def main():
    parser = argparse.ArgumentParser(description="RPi 프로젝트 검색")
    parser.add_argument("query", help="한국어 자연어 질문")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top_k)
    for r in results:
        print(f"[{r['score']:.3f}] {r['title']} ({r['category']}, {r['difficulty']})")
        print(f"  {r['description']}")
        print(f"  {r['url']}")
        print()


if __name__ == "__main__":
    main()
