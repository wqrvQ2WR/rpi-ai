# RPi 용도 추천 AI — Phase 1 데이터 수집

라즈베리파이 프로젝트/OS 추천 데이터셋. 계획서(`계획.md` 등)의 Phase 1 산출물.

## 실행

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
