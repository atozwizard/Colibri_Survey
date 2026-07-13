# [스냅샷] 25GB RAM으로 744B AI 모델을 실행하다, colibrì가 제시하는 새로운 LLM 추론 방식

> 출처: 파파누보, digitalbourgeois.tistory.com/m/3364 (2026-07-13 취득)
> 아래는 원문 요지의 발췌·정리 스냅샷이다. 저작권은 원저자에게 있으며 조사 참고용으로만 보관한다.

## 핵심 주장
- colibrì는 GLM-5.2 744B MoE 모델을 약 25GB RAM 소비자용 PC에서 실행 가능하도록 만든 C 기반 추론 엔진.
- 모델 전체를 메모리에 올리는 대신, 필요한 Expert만 디스크에서 실시간 스트리밍.
- 런타임은 단일 C 소스 + 소규모 헤더로 구성. Python/BLAS/GPU 필수 아님(GPU는 선택).

## MoE 구조를 활용한 메모리 최적화
- GLM-5.2는 744B이나 토큰당 실제 활성 파라미터는 약 40B.
- Dense 영역(Attention, Shared Expert, Embedding)은 RAM 상주.
- Routed Expert는 SSD 저장 후 필요 시 스트리밍.
- Layer별 LRU Cache, OS Page Cache를 추가 캐시 계층으로 활용.

## 디스크 스트리밍 기반 추론
- 21,000개 이상의 Routed Expert를 SSD에 저장하고 필요한 것만 로드.
- 최적화: Layer 단위 LRU Cache, Hot Expert Pinning, OS Page Cache, 비동기 Expert Prefetch, Router 기반 다음 Layer 예측 Prefetch.

## 추론 최적화 기술
- Compressed KV Cache (MLA): 토큰당 저장 데이터 대폭 감소.
- Native MTP Speculative Decoding: MTP Head는 int8 권장.
- DSA Sparse Attention: 필요한 Key만 선택 참조.
- Integer Quantization: int8/int4/int2 + AVX2 정수 dot 커널.

## 메모리/저장 구성
| 구성 요소 | 특징 |
| --- | --- |
| Dense 영역 | RAM 상주 |
| Routed Expert | SSD 저장 후 스트리밍 |
| Expert Cache | RAM 기반 LRU |
| KV Cache | 압축 저장 |
| Hot Expert | RAM/VRAM 유지 |

- 약 370GB int4 모델을 저장 공간에 두고 작은 메모리에서 실행.

## 부가 기능
- 모델 변환 도구, 자동 메모리 계획(Plan), 실행 환경 점검(Doctor), OpenAI 호환 API 서버, Web UI, Windows 11 네이티브, CUDA 선택 가속.

## 결론
- 속도는 최신 GPU 대비 빠르지 않고, 초기에는 SSD 성능 영향이 큼.
- 캐시가 형성되고 Hot Expert가 메모리에 유지되면 개선.
- 목표는 최고 속도가 아니라 "일반 하드웨어에서 초거대 MoE 실행 가능성" 제시.
