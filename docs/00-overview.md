# 00 · 개요 (Overview)

## 요약 (3줄)
- **colibrì**는 744B 규모의 GLM-5.2 MoE 모델을 약 25GB RAM 소비자용 PC에서 구동하는 **순수 C 추론 엔진**이다.
- 핵심 발상은 "모델 전체를 메모리에 올리지 않고, **Dense 부분만 RAM 상주 + Routed Expert는 SSD에서 스트리밍**"하는 것.
- 속도(H100급)가 아니라 **가능성**(값싼 하드웨어에서 프론티어급 모델을 정확히 구동)에 초점을 둔 프로젝트다.

## 배경 / 문제의식
- LLM이 커지면서 수백억~수천억 파라미터 모델은 수백 GB의 메모리 + 고성능 GPU가 필요하다는 인식이 강했다.
- 그러나 MoE(Mixture-of-Experts)는 **토큰 하나 생성 시 전체 파라미터의 일부만 활성화**된다.
  - GLM-5.2(744B)는 토큰당 약 40B만 활성화되고, 그중 토큰마다 바뀌는 routed expert는 약 11GB 정도다.
- colibrì는 이 특성을 이용해 "안 쓰는 파라미터는 디스크에 두고, 필요할 때만 읽는다"는 구조를 택했다.

## colibrì의 메모리 계층
| 구성 요소 | 위치 | 근거 |
|---|---|---|
| Dense (attention, shared expert, embedding, ~17B) | **RAM 상주** (int4, ~9.9GB) | `external/colibri/README.md:19` |
| Routed Expert (21,504개, 75 MoE 레이어 × 256) | **SSD 저장 (~370GB) → 스트리밍** | `external/colibri/README.md:20` |
| Expert Cache | RAM 기반 **레이어별 LRU** | `external/colibri/c/glm.c:1334`(캐시 조회), `:1388`(LRU 승격) |
| Hot Expert | RAM/VRAM **pin** | `external/colibri/c/glm.c:2392`(pin_wire), `:2409`(pin_load) |
| KV Cache | **MLA 압축** 저장 | `external/colibri/c/glm.c:1130`(Lc/Rc 저장) |
| OS Page Cache | 무료 L2 캐시 | `external/colibri/README.md:20` |

## 이 서베이가 다루는 4개 문서
- [`10-colibri-architecture.md`](./10-colibri-architecture.md) — colibri 코드 구조 전반
- [`20-moe-streaming.md`](./20-moe-streaming.md) — MoE 디스크 스트리밍(핵심)
- [`21-mla-kv-compression.md`](./21-mla-kv-compression.md) — MLA 기반 KV Cache 압축
- [`22-speculative-decoding.md`](./22-speculative-decoding.md) — MTP 기반 Speculative Decoding

## 성능에 대한 정직한 시각
- 개발 머신(WSL2, 12코어, 25GB RAM, ~1GB/s NVMe) 기준 **cold ~0.05–0.1 tok/s**. 느리다.
- 병목은 대개 **디스크 대역폭**과 **작은 RAM으로 인한 캐시 용량 제한**이다.
- RAM이 크고 NVMe가 빠를수록(또는 hot expert pin) 빨라진다. 커뮤니티 측정치는 Apple M5 Max에서 ~1–2 tok/s.
  - 근거: `external/colibri/README.md:44`(Honest numbers), `:377`(Community benchmarks).

## 출처
- colibri 원본: https://github.com/JustVugg/colibri (커밋 `5254470`, 2026-07-13)
- 소개 블로그: 파파누보, "25GB RAM으로 744B AI 모델을 실행하다" (2026-07-13) — `../data/colibri/SOURCE.md` 참조
