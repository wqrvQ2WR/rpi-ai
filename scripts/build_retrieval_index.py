#!/usr/bin/env python3
"""RAG용 검색 인덱스 구축. 데이터셋(영문 title/description 위주)을 다국어 임베딩 모델로
인코딩해서 한국어 질문으로도 의미 기반 검색이 되게 한다."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# e5 계열은 검색(retrieval) 태스크 전용으로 학습돼서 일반 문장유사도 모델(MiniLM)보다
# 한국어 질문 <-> 영문 프로젝트 설명 매칭이 눈에 띄게 잘 됨. "query: "/"passage: " 접두어 필수.
MODEL_NAME = "intfloat/multilingual-e5-small"


def entry_to_text(entry: dict) -> str:
    hardware = ", ".join(entry.get("hardware", []))
    use_case = ", ".join(entry.get("use_case", []))
    return (
        f"passage: {entry['title']}. {entry['description']} "
        f"카테고리: {entry['category']}. 난이도: {entry['difficulty']}. "
        f"하드웨어: {hardware}. 키워드: {use_case}."
    )


def main():
    entries = json.loads((DATA_DIR / "rpi_projects_dataset.json").read_text(encoding="utf-8"))
    texts = [entry_to_text(e) for e in entries]

    print(f"loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"encoding {len(texts)} entries...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    np.save(DATA_DIR / "retrieval_embeddings.npy", embeddings)
    (DATA_DIR / "retrieval_metadata.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )
    (DATA_DIR / "retrieval_model.txt").write_text(MODEL_NAME, encoding="utf-8")

    print(f"saved embeddings: {embeddings.shape} -> data/retrieval_embeddings.npy")
    print("saved metadata -> data/retrieval_metadata.json")


if __name__ == "__main__":
    main()
