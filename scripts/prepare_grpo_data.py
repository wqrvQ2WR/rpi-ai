#!/usr/bin/env python3
"""Phase 4 GRPO 학습용 프롬프트 데이터셋 생성.

SFT 학습 데이터의 human_query들을 질문 삼아 각각에 대해 search.py로 실제 검색 후보를
붙인다. GRPOTrainer는 "prompt" 컬럼(채팅 메시지 리스트)만 있으면 되고, reward function이
검색 후보와 대조해서 채점해야 하므로 "candidates"를 JSON 문자열 컬럼으로 같이 저장한다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import search  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = (
    "당신은 라즈베리파이(Raspberry Pi) 프로젝트를 추천해주는 AI 어시스턴트입니다. "
    "아래 [후보 목록]에 있는 항목 중에서만 골라서 추천하세요. 목록에 없는 프로젝트나 "
    "URL을 절대로 지어내지 마세요. 후보 중 적절한 게 없으면 없다고 솔직히 답하세요. "
    "한국어로, 왜 추천하는지와 참고 링크를 함께 답하세요."
)

TOP_K = 5
MAX_PROMPTS = 150  # DeepSeek 채점 비용/속도 고려해서 제한


def build_user_content(query: str, candidates: list) -> str:
    lines = ["[후보 목록]"]
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. {c['title']} ({c['category']}, 난이도: {c['difficulty']}) - "
            f"{c['description']} (링크: {c['url']})"
        )
    lines.append("")
    lines.append(f"[질문]\n{query}")
    return "\n".join(lines)


def main():
    train_rows = [
        json.loads(line)
        for line in (DATA_DIR / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = [row["messages"][1]["content"] for row in train_rows][:MAX_PROMPTS]

    out_rows = []
    for query in queries:
        candidates = search(query, top_k=TOP_K)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(query, candidates)},
        ]
        out_rows.append({
            "prompt": prompt,
            "candidates": json.dumps(candidates, ensure_ascii=False),
        })

    out_path = DATA_DIR / "grpo_prompts.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(out_rows)}개 프롬프트 -> {out_path}")


if __name__ == "__main__":
    main()
