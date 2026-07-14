# 90 · 서베이 상태 · 마감 · 재개 안내

colibri 서베이의 최종 상태를 정리한다. **문서/코드 산출물은 완결**되었고, 유일하게 남은 것은 *선택적* 서버 성능 실측(정성 주장에 실수치를 채우는 보강)이다.

## 1. 완료 상태 (Status)

| 영역 | 산출물 | 상태 |
|---|---|---|
| 개요·아키텍처 | `00`, `10` | ✅ 완료 |
| 핵심 기술(MoE 스트리밍·MLA·speculative) | `20`, `21`, `22` | ✅ 완료 |
| 로컬 빌드/실행 | `30` | ✅ 완료 |
| **엔진 정확성 실검증** | `31` | ✅ **실행됨**(C테스트 4종 + tiny GLM oracle 32/32) |
| 분석(tradeoff·자원·타모델) | `40`, `50`, `60`, `61`, `62` | ✅ 완료 |
| 경영·기술 통합 브리프 | `70` | ✅ 완료 |
| OLMoE 위상 · H100 추천 | `80` | ✅ 완료 |
| **(c) ThinkFlow 업그레이드 설계+런북** | `81`, `84` | ✅ 완료(실행은 운영자) |
| **(a) MXFP4→int4 변환기** | `82` + `scripts/mxfp4_to_int4_prototype.py` | ✅ 레이아웃 검증+selftest PASS |
| **(b) 스트리밍 실측** | `83` + `scripts/olmoe_streaming_bench.sh` | 🟡 로컬 정확성 증명 완료 / **서버 성능수치 보류** |

## 2. 검증으로 증명된 사실 (근거 있는 결론)
1. **엔진 정확성**: `glm.c`가 transformers GlmMoeDsa 오라클과 token-exact(32/32). (`31`)
2. **스트리밍 투명성**: 캐시 cap을 8→1로 줄여도 출력 불변(32/32) → LRU 축출/재적재는 정확성에 무해. (`83 §1`)
3. **gpt-oss 변환 경로**: 실제 MXFP4 레이아웃(expert MLP만, grouped `[E,O,90,16]`) 확인 + int4 변환 수학 검증(round-trip rel-err 0.071). (`82`, `data/topics/apply-gpt-oss/weight-layout-verified.md`)
4. **적용 판단**: colibri는 "VRAM 초과" 영역 전용 → ThinkFlow(H100에 들어가는 모델)엔 부적합, 대신 서빙 모델 업그레이드가 정답. (`80`, `81`)

## 3. 보류된 유일한 항목: 서버 성능 실측 (선택)
- **무엇**: OLMoE 실가중치로 cap↔hit-rate↔tok/s↔expert-disk **성능 곡선** 수집.
- **왜 보류**: 필수 아님. 서베이 결론은 이미 확립. 이는 `40`/`50`의 정성 표를 정량화하는 보강일 뿐.
- **왜 쉬움(재개 시)**: colibri 벤치는 **GPU 무관**(CPU+디스크+RAM만). ThinkFlow 서버는 CPU 97% idle·RAM 228GB 여유 → GPU가 꽉 차 있어도 무방해.

### 재개 절차 (원할 때)
```bash
# ThinkFlow H100 박스에서 한 줄(격리 내장: 2~4코어, 저 IO 우선순위):
git clone https://github.com/atozwizard/Colibri_Survey && cd Colibri_Survey
CORES=0-3 MEM=8G bash scripts/olmoe_streaming_bench.sh
# 결과(cap, hit_rate, tok/s, expert-disk, RSS)를 docs/40·50 표에 추기.
```

## 4. (1) ThinkFlow 스왑 — 운영자 실행 대기
- 실행 가능 상태로 확정됨: 런북 `docs/84` + 스크립트 `scripts/thinkflow_swap_rehearsal.sh`.
- 권장 순서: `--preflight`(무중단, 지금) → 유지보수 창에서 `--cutover` → KPI A/B → 확정/롤백.

## 5. 저장소 지도 (요약)
- `docs/` 00~90: 분석·설계·검증 문서.
- `scripts/`: 변환기 프로토타입 / 스트리밍 벤치 / 스왑 런북 스크립트.
- `data/`: GLM·colibri·토픽별·OLMoE·gpt-oss 레이아웃 근거 자료.
- `external/colibri/`: 벤더링된 원본 소스(+ 로컬 빌드 산출물은 `.gitignore`).

## 6. 결론
서베이는 **문서·코드·로컬검증 기준으로 완결**되었다. 서버 성능 실측과 120b 스왑은 *운영 실행 단계*로, 준비물(스크립트·런북·프로토콜)이 모두 갖춰져 필요 시 즉시 재개 가능하다.
