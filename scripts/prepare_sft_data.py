#!/usr/bin/env python3
"""Phase 2 SFT용 데이터 변환: rpi_projects_dataset.json -> Qwen2.5 chat 포맷 JSONL.
TRL SFTTrainer가 바로 읽을 수 있는 conversational 포맷({"messages": [...]})으로 저장한다."""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = (
    "당신은 라즈베리파이(Raspberry Pi) 프로젝트와 용도를 추천해주는 AI 어시스턴트입니다. "
    "사용자의 상황과 목적에 맞는 프로젝트, 필요한 하드웨어, 난이도를 함께 안내하세요."
)

VAL_RATIO = 0.1
SEED = 42


def to_messages(entry: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry["human_query"]},
            {"role": "assistant", "content": entry["ai_expected_output"]},
        ]
    }


def main():
    entries = json.loads((DATA_DIR / "rpi_projects_dataset.json").read_text(encoding="utf-8"))
    rows = [to_messages(e) for e in entries]

    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    train_path = DATA_DIR / "sft_train.jsonl"
    val_path = DATA_DIR / "sft_val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train: {len(train_rows)} -> {train_path}")
    print(f"val: {len(val_rows)} -> {val_path}")


if __name__ == "__main__":
    main()
