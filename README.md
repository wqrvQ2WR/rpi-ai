# RPi 용도 추천 AI

라즈베리파이 프로젝트/OS를 추천해주는 AI. 오픈모델(Qwen2.5-1.5B) + SFT + RL + 구글 무료
리소스로 $0 운영을 목표로 함.

## Phase 1 — 데이터 수집

### 실행

```bash
gh api orgs/raspberrypilearning/repos --paginate -q '.[] | select(.description != null) | [.name, (.archived|tostring), .description] | @tsv' > data/raspberrypilearning_repos_all.tsv
curl -s https://raw.githubusercontent.com/thibmaek/awesome-raspberry-pi/master/README.md -o data/awesome_raspberry_pi_raw.md
python3 scripts/build_dataset.py
```

결과: `data/rpi_projects_dataset.json`

## 수집 소스

| 소스 | 항목 수 | 방식 |
|------|--------|------|
| raspberrypilearning (GitHub org, 공식 프로젝트) | 219 | `gh api` repo 목록 (name/description) |
| awesome-raspberry-pi (Projects/Tools 섹션) | 145 | README 마크다운 파싱 |
| awesome-raspberry-pi (OS Images 섹션) | 62 | README 마크다운 파싱, category="OS/펌웨어" |
| **합계 (중복 제거)** | **426** | |

Hackaday.io API는 dev.hackaday.io 계정으로 API 키를 발급받아야 해서 이번 자동 수집에서는 제외했음. Reddit r/raspberry_pi도 OAuth 앱 등록이 필요해서 제외 — 필요하면 크레덴셜 설정 후 추가 가능.

## 스키마

계획서의 `human_query` / `ai_expected_output` 필드는 title/description으로부터 **템플릿 자동 생성**한 것이라 표현이 기계적임 (예: "X를(을) 추천합니다..."). category/difficulty/hardware도 키워드 휴리스틱이라 완벽하지 않음. Phase 3에서 계획된 Gemini API 평가 패스를 활용해 한국어 표현 다듬기 + 라벨 재검증을 하면 품질이 올라감.

## 알려진 한계

- 426개 (계획서 목표 500+에 약간 못 미침) — Reddit/Hackaday/Instructables 추가하면 채울 수 있음
- "기타" 카테고리가 136개(32%)로 큼 — 키워드 룰 확장 여지 있음
- human_query/ai_expected_output이 영문 제목/설명 그대로 들어가 있어 완전한 한국어 문장이 아님

## Phase 2 — SFT

`scripts/prepare_sft_data.py`로 Phase 1 데이터셋을 Qwen2.5 채팅 포맷(JSONL, `messages` 컬럼)으로
변환해 `data/sft_train.jsonl`(384개) / `data/sft_val.jsonl`(42개)로 분리.

```bash
python3 scripts/prepare_sft_data.py
```

학습은 로컬 GPU가 없어서 [notebooks/phase2_sft_colab.ipynb](notebooks/phase2_sft_colab.ipynb)를
Google Colab(T4 GPU, 무료)에서 직접 실행. 저장소가 공개라 아래 링크로 바로 열림:

https://colab.research.google.com/github/wqrvQ2WR/rpi-ai/blob/main/notebooks/phase2_sft_colab.ipynb

노트북 내용: `Qwen/Qwen2.5-1.5B-Instruct`를 LoRA(peft) + TRL `SFTTrainer`로 fp16 학습(T4는
bf16 텐서코어가 없어서 fp16 사용) → LoRA 어댑터 저장 → 베이스 모델과 병합 → 간단 추론 확인 →
결과 zip 다운로드. GGUF 변환(Phase 5)은 이 병합된 모델을 입력으로 별도 진행.
