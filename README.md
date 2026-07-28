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
| raspberrypilearning (GitHub org, 공식 프로젝트) | 158 | `gh api` repo 목록 (name/description), Scratch 코딩연습/admin 성격 repo 제외 |
| awesome-raspberry-pi (Projects/Tools 섹션) | 145 | README 마크다운 파싱 |
| awesome-raspberry-pi (OS Images 섹션) | 62 | README 마크다운 파싱, category="OS/펌웨어" |
| **합계 (중복 제거)** | **365** | |

Hackaday.io API는 dev.hackaday.io 계정으로 API 키를 발급받아야 해서 이번 자동 수집에서는 제외했음. Reddit r/raspberry_pi도 OAuth 앱 등록이 필요해서 제외 — 필요하면 크레덴셜 설정 후 추가 가능.

## 스키마

계획서의 `human_query` / `ai_expected_output` 필드는 title/description으로부터 **템플릿 자동 생성**한 것이라 표현이 기계적임 (예: "X를(을) 추천합니다..."). category/difficulty/hardware도 키워드 휴리스틱이라 완벽하지 않음. Phase 3에서 계획된 Gemini API 평가 패스를 활용해 한국어 표현 다듬기 + 라벨 재검증을 하면 품질이 올라감.

## 알려진 한계

- 365개 (계획서 목표 500+에 못 미침) — Reddit/Hackaday/Instructables 추가하면 채울 수 있음
- "기타" 카테고리가 83개(23%)로 여전히 큼 — 키워드 룰 확장 여지 있음
- human_query/ai_expected_output이 영문 제목/설명 그대로 들어가 있어 완전한 한국어 문장이 아님

## Phase 2 — SFT

`scripts/prepare_sft_data.py`로 Phase 1 데이터셋을 Qwen2.5 채팅 포맷(JSONL, `messages` 컬럼)으로
변환해 `data/sft_train.jsonl`(329개) / `data/sft_val.jsonl`(36개)로 분리.

```bash
python3 scripts/prepare_sft_data.py
```

학습은 로컬 GPU가 없어서 [notebooks/phase2_sft_colab.ipynb](notebooks/phase2_sft_colab.ipynb)를
Google Colab(T4 GPU, 무료)에서 직접 실행. 저장소가 공개라 아래 링크로 바로 열림:

https://colab.research.google.com/github/wqrvQ2WR/rpi-ai/blob/main/notebooks/phase2_sft_colab.ipynb

노트북 내용: `Qwen/Qwen2.5-1.5B-Instruct`를 LoRA(peft) + TRL `SFTTrainer`로 fp16 학습(T4는
bf16 텐서코어가 없어서 fp16 사용) → LoRA 어댑터 저장 → 베이스 모델과 병합 → 간단 추론 확인 →
결과를 Google Drive에 저장(zip 다운로드는 큰 파일이 끊기기 쉬워서 비권장). GGUF 변환(Phase 5)은
이 병합된 모델을 입력으로 별도 진행.

**실제 학습 결과 (2026-07-28)**: 답변 포맷("OO를(을) 추천합니다 + 설명 + 참고 링크")은 잘
학습됐지만, 384개 정도의 작은 데이터로 몇 epoch만 돌린 1.5B 모델은 실제로 존재하지 않는
프로젝트 제목/GitHub URL을 지어내는(hallucination) 문제가 있었음. 예: 실제 데이터셋에 없는
`no-advertising-connection`이라는 가짜 repo를 만들어냄. → 아래 검색(RAG) 단계로 보완.

## 검색 (RAG) — 진짜 링크만 추천하기

모델이 URL을 통째로 암기하게 하는 대신, 365개 데이터셋을 임베딩 인덱스로 만들어두고
한국어 질문으로 의미 기반 검색을 해서 **실제로 존재하는 항목만** 후보로 준다. 모델은 이
후보를 자연스러운 문장으로 설명하는 역할만 하면 되므로 링크 할루시네이션이 구조적으로
불가능해진다.

```bash
pip install -r requirements.txt
python3 scripts/build_retrieval_index.py   # data/rpi_projects_dataset.json -> 임베딩 인덱스
python3 scripts/search.py "집에서 광고 없이 인터넷 쓰고 싶어" --top-k 5
```

임베딩 모델은 `intfloat/multilingual-e5-small` (검색 특화, 한국어 질문 <-> 영문 프로젝트
설명 매칭이 일반 문장유사도 모델보다 훨씬 정확함 — `query: ` / `passage: ` 접두어 필요).
결과: `data/retrieval_embeddings.npy` (365 x 384 float32), `data/retrieval_metadata.json`.

이 검색 품질을 확인한 뒤, 필요하면 "검색된 실제 항목 + 질문 → 그 항목을 설명하는 답변"
형태로 SFT 데이터를 다시 만들어 재학습하는 게 다음 단계 (아직 미착수).
