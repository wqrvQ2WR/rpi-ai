#!/usr/bin/env python3
"""Phase 3 배치 평가: 여러 질문에 대해 (검색 -> 로컬 Ollama 생성 -> reward model 채점)
파이프라인을 전부 돌려서 평균 리워드를 낸다. 계획서의 "Reward 점수 평균 2.0/3.0 이상"
목표와 비교하기 위한 스크립트."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_test import call_ollama  # noqa: E402
from reward_model import reward  # noqa: E402
from search import search  # noqa: E402

TEST_QUERIES = [
    "집에서 광고 없이 인터넷 쓰고 싶어",
    "NAS 만들고 싶은데 어떤 라즈베리파이가 좋을까?",
    "집 밖에서도 안전하게 집 네트워크에 접속하고 싶어",
    "라즈베리파이로 로봇 만들고 싶어",
    "홈시어터/미디어센터 만들고 싶어",
    "아이한테 코딩 가르치고 싶어",
    "식물에 자동으로 물 주는 시스템 만들고 싶어",
    "라즈베리파이로 날씨 관측하고 싶어",
]


def main():
    results = []
    for query in TEST_QUERIES:
        candidates = search(query, top_k=5)
        answer = call_ollama(query, candidates)
        r = reward(query, candidates, answer)
        results.append({"query": query, "answer": answer, **r})

        print(f"[{r['score']:+d}] {query}")
        print(f"  답변: {answer[:120].replace(chr(10), ' ')}...")
        print(f"  이유: {r['reason']}")
        print()

    avg = sum(r["score"] for r in results) / len(results)
    print(f"=== 평균 리워드: {avg:.2f} / 3.0 (계획서 목표: 2.0 이상) ===")

    out_path = Path(__file__).resolve().parent.parent / "data" / "phase3_eval_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
