# Colibri Survey

[colibrì](https://github.com/JustVugg/colibri) — GLM-5.2 (744B MoE) 모델을 약 25GB RAM 소비자용 PC에서 구동하는
순수 C 기반 디스크 스트리밍 추론 엔진 — 과 그 핵심 기술에 대한 서베이 저장소.

## 조사 주제
- **colibri 아키텍처**: Dense 상주 + Routed Expert의 SSD 스트리밍 구조
- **MoE 디스크 스트리밍**: Expert offloading, LRU 캐시, prefetch
- **MLA KV Cache 압축**: 긴 컨텍스트에서의 메모리 절감
- **Speculative Decoding**: MTP(Multi-Token Prediction) 기반 추론 가속

## 저장소 구조
```text
├── AGENTS.md      # 서베이 작업 규칙 · 작성 방향
├── external/
│   └── colibri/   # 원본 소스 vendored (참조 전용)
├── docs/          # 우리가 작성하는 분석 문서
└── data/          # 수집한 원자료
    ├── glm/
    ├── colibri/
    └── topics/{moe-streaming, mla-kv, speculative-decoding}/
```

## 문서 목차 (docs/)
- `00-overview.md` — 전체 개요
- `10-colibri-architecture.md` — colibri 코드 구조 분석
- `20-moe-streaming.md` — MoE 디스크 스트리밍
- `21-mla-kv-compression.md` — MLA KV 압축
- `22-speculative-decoding.md` — Speculative Decoding
- `30-local-run-notes.md` — 로컬 빌드/실행(coli doctor) 노트
- `31-engine-verification.md` — 엔진 정확성 실검증(C 테스트 + tiny GLM oracle 32/32)
- `40-analysis-tradeoffs.md` — 논리적 장점·단점·trade-off 분석
- `50-resource-requirements.md` — 필요 자원(하드웨어 등급별)
- `60-applying-to-other-models.md` — 타 모델 적용 방안(일반)
- `61-apply-gpt-oss-20b.md` — gpt-oss-20b 적용 설계서
- `62-apply-gemma4.md` — gemma4 적용 설계서(적합성 판정 포함)
- `70-executive-brief.md` — 경영·기술 통합 브리프
- `80-olmoe-and-h100-recommendations.md` — OLMoE 위상 · H100 서빙 추천 모델
- `81-thinkflow-upgrade-design.md` — (c) ThinkFlow H100 서빙 모델 업그레이드 설계
- `82-gpt-oss-mxfp4-to-int4-converter.md` — (a) MXFP4→int4 변환기 프로토타입
- `83-olmoe-streaming-measurement.md` — (b) 스트리밍 실측(로컬 실증 + H100 프로토콜)
- `84-thinkflow-swap-checklist.md` — (1) 120b 스왑 운영자 런북/체크리스트
- `85-dual-model-crossval-resources.md` — 2모델 교차검증(골든셋) 자원 분석
- `90-status-and-closeout.md` — 서베이 상태·마감·재개 안내
- `99-references.md` — 참고문헌

## 스크립트 (scripts/)
- `mxfp4_to_int4_prototype.py` — gpt-oss MXFP4→colibri int4 변환기(검증된 레이아웃, `--selftest`)
- `olmoe_streaming_bench.sh` — OLMoE 스트리밍 실측(공유서버 안전: cgroup/nice/ionice/iobench)
- `thinkflow_swap_rehearsal.sh` — ThinkFlow LLM 무중단 스왑 리허설(preflight/cutover/rollback)
- `build_report.py` — 전 문서 결합 + mermaid→이미지 렌더 → `report/master.md`
- `make_reference_docx.py` — 기업용 docx 테마(`report/reference.docx`)

## 종합 보고서 (report/)
`report/Colibri_Survey_Report.docx` — 전 문서를 요약 없이 결합한 DOCX(목차·표·다이어그램 이미지·기업 테마). 재생성:
```bash
uv run --with python-docx python scripts/make_reference_docx.py
uv run --with python-docx python scripts/build_report.py   # mermaid 렌더(mmdc 필요, 없으면 코드블록 유지)
pandoc report/master.md -o report/Colibri_Survey_Report.docx --toc --toc-depth=3 \
  --reference-doc=report/reference.docx --resource-path=report
```

## 빌드/실행 (uv)
의존성은 [uv](https://docs.astral.sh/uv/)로 관리한다. C 엔진과 `coli` CLI는 표준 라이브러리만 쓰므로 추가 설치가 필요 없다.
```bash
uv venv --python 3.14
make -C external/colibri/c glm                              # 엔진 빌드
uv run --no-sync python external/colibri/c/coli doctor      # 준비 상태 점검
```
자세한 로컬 노트는 [`docs/30-local-run-notes.md`](./docs/30-local-run-notes.md) 참조.

## 규칙
작업 규칙과 작성 방향은 [`AGENTS.md`](./AGENTS.md) 참조.

## 라이선스 / 출처
`external/colibri/`는 원저작자([JustVugg/colibri](https://github.com/JustVugg/colibri))의 라이선스를 따른다.
`data/`의 수집 자료는 각 디렉토리의 `SOURCE.md`에 출처를 명시한다.
