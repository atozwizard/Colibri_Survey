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
- `99-references.md` — 참고문헌

## 규칙
작업 규칙과 작성 방향은 [`AGENTS.md`](./AGENTS.md) 참조.

## 라이선스 / 출처
`external/colibri/`는 원저작자([JustVugg/colibri](https://github.com/JustVugg/colibri))의 라이선스를 따른다.
`data/`의 수집 자료는 각 디렉토리의 `SOURCE.md`에 출처를 명시한다.
