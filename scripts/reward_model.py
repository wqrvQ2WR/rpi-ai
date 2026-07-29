#!/usr/bin/env python3
"""Phase 3 Reward Model.

계획서는 Gemini API로 답변 품질을 평가하는 걸 가정했지만, 이번엔 DeepSeek API(OpenAI 호환)로
대체해서 같은 역할을 한다. 점수 체계는 계획서 그대로:
  +3 완벽한 추천 / +1 적절한 추천 / -1 엉뚱한 추천 / -3 잘못된 정보

RAG 구조라서 "잘못된 정보"의 가장 심각한 형태는 검색 후보에 없는 URL을 지어내는 것이고,
이건 LLM 판정 없이 규칙(rule-based)으로 100% 정확하게 잡을 수 있다. 그래서:
  1. grounding check (무료, 결정적): 답변에 나온 URL이 후보 목록에 없으면 무조건 -3
  2. grounding을 통과하면 DeepSeek을 판정자로 써서 정확성/유용성/실용성 채점
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

JUDGE_SYSTEM_PROMPT = """당신은 라즈베리파이 프로젝트 추천 AI의 답변 품질을 채점하는 평가자입니다.
아래 [질문], [실제 후보 목록](이 안의 정보만 사실로 인정), [AI의 답변]을 보고 채점하세요.

점수 기준 (정수 하나만):
+3: 후보 중 가장 적절한 걸 골라서 정확하고 실용적으로 설명함
+1: 대체로 적절하지만 설명이 부실하거나 최선의 선택은 아님
-1: 후보와 동떨어지거나 질문 의도를 잘못 이해함
-3: 후보 목록에 없는 사실을 지어내거나 완전히 틀린 정보를 줌

반드시 JSON만 출력하세요: {"score": <정수>, "reason": "<한 문장 이유, 한국어>"}"""


def _call_deepseek(system: str, user: str) -> str:
    if not API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY가 없습니다 (.env 확인)")
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def grounding_check(answer: str, candidates: list) -> tuple:
    candidate_urls = {c["url"] for c in candidates}
    # 마크다운 링크 [text](url) 뒤에 공백 없이 텍스트가 바로 붙거나, 링크가 중첩/깨진
    # 경우가 흔해서 공백뿐 아니라 ) ] [ 도 URL 종료 문자로 취급
    # (실제 후보 URL 중에 이 문자들이 들어간 건 없음)
    found_urls = set(re.findall(r"https?://[^\s)\]\[]+", answer))
    found_urls = {u.rstrip(".,，。") for u in found_urls}
    bad_urls = [u for u in found_urls if u not in candidate_urls]
    return (len(bad_urls) == 0, bad_urls)


def llm_judge(query: str, candidates: list, answer: str) -> dict:
    candidate_text = "\n".join(
        f"- {c['title']}: {c['description']} (링크: {c['url']})" for c in candidates
    )
    user_prompt = f"[질문]\n{query}\n\n[실제 후보 목록]\n{candidate_text}\n\n[AI의 답변]\n{answer}"
    raw = _call_deepseek(JUDGE_SYSTEM_PROMPT, user_prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"score": 0, "reason": f"판정 파싱 실패: {raw[:100]}"}
    json_text = re.sub(r'"score"\s*:\s*\+', '"score": ', match.group(0))
    try:
        parsed = json.loads(json_text)
        return {"score": int(parsed["score"]), "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0, "reason": f"판정 파싱 실패: {raw[:100]}"}


def reward(query: str, candidates: list, answer: str) -> dict:
    ok, bad_urls = grounding_check(answer, candidates)
    if not ok:
        return {
            "score": -3,
            "reason": f"후보에 없는 URL을 지어냄: {bad_urls}",
            "grounding_ok": False,
        }
    judged = llm_judge(query, candidates, answer)
    judged["grounding_ok"] = True
    return judged


def main():
    if len(sys.argv) < 2:
        print("usage: reward_model.py <query> [--fake-url]")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from search import search

    query = sys.argv[1]
    candidates = search(query, top_k=5)

    if "--fake-url" in sys.argv:
        answer = (
            f"{candidates[0]['title']}을(를) 추천합니다. "
            "자세한 내용은 https://github.com/totally-fake-repo/does-not-exist 를 참고하세요."
        )
        print("(테스트용으로 존재하지 않는 URL을 넣은 답변)")
    else:
        c = candidates[0]
        answer = f"{c['title']}을(를) 추천합니다. {c['description']} 참고: {c['url']}"

    print(f"질문: {query}")
    print(f"답변: {answer}\n")
    result = reward(query, candidates, answer)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
