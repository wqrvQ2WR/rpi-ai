#!/usr/bin/env python3
"""RAG 검색: 한국어 자연어 질문 -> 데이터셋에서 실제로 존재하는 top-k 프로젝트 반환.
build_retrieval_index.py를 먼저 실행해서 임베딩을 만들어둬야 한다.

의미 기반 검색(e5 임베딩) + 키워드 검색(BM25)을 RRF(reciprocal rank fusion)로 결합한
하이브리드 방식. "NAS", "VPN", "Docker" 같은 영문 약어/고유명사는 다국어 임베딩 모델이
의미를 잘 못 잡아서 순위가 밀리는 경우가 있는데, BM25가 텍스트에 그 단어가 그대로
있으면 정확히 잡아내서 보완해준다. 순수 한국어 질문(약어가 없는 경우)은 BM25가 거의
기여하지 않고 임베딩 점수가 자연히 그대로 반영된다."""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RRF_K = 60


def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _load():
    model_name = (DATA_DIR / "retrieval_model.txt").read_text(encoding="utf-8").strip()
    model = SentenceTransformer(model_name)
    embeddings = np.load(DATA_DIR / "retrieval_embeddings.npy")
    metadata = json.loads((DATA_DIR / "retrieval_metadata.json").read_text(encoding="utf-8"))

    bm25_corpus = [
        _tokenize(f"{m['title']} {m['description']}") for m in metadata
    ]
    bm25 = BM25Okapi(bm25_corpus)
    return model, embeddings, metadata, bm25


def search(query: str, top_k: int = 5):
    model, embeddings, metadata, bm25 = _load()

    query_emb = model.encode(["query: " + query], normalize_embeddings=True).astype("float32")[0]
    semantic_scores = embeddings @ query_emb
    semantic_rank = np.argsort(-semantic_scores)

    bm25_scores = np.array(bm25.get_scores(_tokenize(query)))
    bm25_rank = np.argsort(-bm25_scores)

    rrf_scores = np.zeros(len(metadata))
    for rank, idx in enumerate(semantic_rank):
        rrf_scores[idx] += 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(bm25_rank):
        if bm25_scores[idx] <= 0:
            continue
        rrf_scores[idx] += 1.0 / (RRF_K + rank)

    top_idx = np.argsort(-rrf_scores)[:top_k]
    return [
        {
            "score": float(rrf_scores[i]),
            "semantic_score": float(semantic_scores[i]),
            "bm25_score": float(bm25_scores[i]),
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
    parser = argparse.ArgumentParser(description="RPi 프로젝트 검색 (하이브리드: 임베딩 + BM25)")
    parser.add_argument("query", help="한국어 자연어 질문")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="세부 점수(semantic/bm25) 표시")
    args = parser.parse_args()

    results = search(args.query, args.top_k)
    for r in results:
        if args.verbose:
            print(
                f"[rrf {r['score']:.4f} | sem {r['semantic_score']:.3f} | "
                f"bm25 {r['bm25_score']:.2f}] {r['title']} ({r['category']}, {r['difficulty']})"
            )
        else:
            print(f"[{r['score']:.4f}] {r['title']} ({r['category']}, {r['difficulty']})")
        print(f"  {r['description']}")
        print(f"  {r['url']}")
        print()


if __name__ == "__main__":
    main()
