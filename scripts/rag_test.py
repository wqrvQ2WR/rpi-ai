#!/usr/bin/env python3
"""RAG 프롬프트 테스트: search.py로 실제 후보를 찾은 다음, 로컬 Ollama 모델에게
"이 후보들 안에서만 골라서 답해"라고 시켜서 할루시네이션 없이 답하는지 확인한다.
(Colab에서 SFT한 모델이 아니라 로컬 Ollama의 베이스 Qwen2.5로 먼저 검증)"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import search  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"

SYSTEM_PROMPT = (
    "당신은 라즈베리파이(Raspberry Pi) 프로젝트를 추천해주는 AI 어시스턴트입니다. "
    "아래 [후보 목록]에 있는 항목 중에서만 골라서 추천하세요. 목록에 없는 프로젝트나 "
    "URL을 절대로 지어내지 마세요. 후보 중 적절한 게 없으면 없다고 솔직히 답하세요. "
    "한국어로, 왜 추천하는지와 참고 링크를 함께 답하세요."
)


def build_prompt(query: str, candidates: list) -> str:
    lines = ["[후보 목록]"]
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. {c['title']} ({c['category']}, 난이도: {c['difficulty']}) - "
            f"{c['description']} (링크: {c['url']})"
        )
    lines.append("")
    lines.append(f"[질문]\n{query}")
    return "\n".join(lines)


def call_ollama(query: str, candidates: list) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, candidates)},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    candidates = search(args.query, args.top_k)
    print("=== 검색된 후보 ===")
    for c in candidates:
        print(f"- {c['title']} ({c['url']})")
    print()

    print("=== 모델 답변 ===")
    answer = call_ollama(args.query, candidates)
    print(answer)

    print()
    print("=== 검증: 답변에 등장한 URL이 후보 안에 있는지 ===")
    candidate_urls = {c["url"] for c in candidates}
    for url in candidate_urls:
        if url in answer:
            print(f"  OK  (후보에 있음) {url}")
    import re
    found_urls = set(re.findall(r"https?://\S+", answer))
    unknown = found_urls - candidate_urls
    for url in unknown:
        print(f"  경고 (후보에 없는 URL 등장) {url.rstrip('.,)')}")


if __name__ == "__main__":
    main()
