# AGENTS.md — Colibri Survey 작업 규칙

이 저장소에서 작업하는 모든 에이전트/기여자는 아래 규칙을 따른다.

## 1. 목적
- **colibrì** (GLM-5.2 744B MoE 모델을 약 25GB RAM 소비자용 PC에서 구동하는 순수 C 추론 엔진)와
  그 핵심 기술을 코드·논문 수준까지 조사·정리하는 **서베이 저장소**.
- 중점 조사 주제 3가지:
  1. MoE 디스크 스트리밍 (Expert offloading / streaming)
  2. MLA 기반 KV Cache 압축
  3. Speculative Decoding (특히 MTP 기반)

## 2. 저장소 구조 규칙
| 경로 | 용도 | 규칙 |
|---|---|---|
| `external/colibri/` | 원본 소스 vendored 복사본 | **직접 수정 금지(참조 전용)**, 원본 LICENSE 유지, `.git` 제거 |
| `docs/` | 우리가 작성하는 분석/서베이 문서 | 파일명 `NN-topic.md` (번호 접두어로 정렬) |
| `data/glm/` | GLM-5.2 / MoE 일반 자료 | 원자료 + `SOURCE.md` |
| `data/colibri/` | colibri 관련 자료(블로그·README 스냅샷 등) | 원자료 + `SOURCE.md` |
| `data/topics/{moe-streaming,mla-kv,speculative-decoding}/` | 토픽별 수집 자료 | 원자료 + `SOURCE.md` |

- **핵심 원칙: 가공물(우리 글)과 원자료(수집물)를 절대 섞지 않는다.** 해석·분석은 `docs/`, 원본 수집물은 `data/`.

## 3. 문서 작성 방향
- 언어: **한국어**.
- 각 `docs/` 문서는 다음 골격을 따른다:
  1. 요약 (3줄)
  2. 배경 / 문제의식
  3. 핵심 메커니즘
  4. 코드/수식 근거 (파일·라인 인용)
  5. 한계 및 트레이드오프
  6. 출처
- 모든 주장에는 근거를 붙인다: `external/colibri` 코드 라인 또는 `data/`의 출처.
- 코드 인용은 `파일경로:라인` 형식으로 추적 가능하게 표기.

## 4. 자료 수집 규칙
- 각 수집 디렉토리에는 `SOURCE.md`를 두고 (출처 URL, 저자, 취득일, 라이선스)를 기록한다.
- **1차 출처(원 저자 repo/논문) 우선**, 2차 출처(블로그·기사)는 보조.
- 저작권이 명확히 재배포 가능한 자료(arXiv 등)는 원문 저장 허용.
- 저작권이 불명확하면 원문 저장 대신 링크 + 요약으로 대체.

## 5. Git 관리 규칙
- 기본 브랜치: `main`.
- 커밋 단위는 작게. 한글 메시지 허용. 형식: `type: 요약`
  - `type` ∈ {feat, docs, data, chore, fix}
  - 예) `data: colibri README 스냅샷 추가`, `docs: MoE 스트리밍 구조 분석 초안`
- 원본 소스 vendoring은 별도 커밋으로 분리.
- 대용량 바이너리(모델 가중치·체크포인트)는 커밋 금지 → `.gitignore` 참조.

## 6. 범위 밖 (하지 않을 것)
- 모델 가중치/체크포인트 저장, 실제 추론 실행 산출물 커밋.
- `external/colibri` 원본 소스 개조.
