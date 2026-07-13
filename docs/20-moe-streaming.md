# 20 · MoE 디스크 스트리밍

## 요약 (3줄)
- MoE는 토큰마다 소수 expert만 활성화되므로, 비활성 expert를 값싼 저장소(RAM/SSD)에 두고 필요할 때만 로드하는 **offloading**이 가능하다.
- colibrì는 이를 극단화해 **21,504개 routed expert(~370GB int4)를 SSD에 두고 스트리밍**하고, Dense는 RAM에 상주시킨다.
- 병목(디스크 I/O)을 줄이기 위해 **레이어별 LRU 캐시 + batch-union + 비동기 prefetch(WILLNEED) + router-lookahead + hot-expert pin + OS page cache**를 결합한다.

## 배경 / 문제의식
- MoE 모델은 총 파라미터는 거대하지만 토큰당 활성 파라미터는 작다(GLM-5.2: 744B 중 ~40B).
- 그래도 "전체 가중치"는 어딘가 저장돼야 한다. GPU VRAM에 다 올리면 비싸다.
- 선행 연구는 비활성 expert를 CPU RAM/SSD로 offload하고 on-demand 로드한다(예: Eliseev & Mazur, arXiv:2312.17238; DALI, arXiv:2602.03495).
  - 공통 난제: **활성 expert는 입력 의존적**이라 라우팅이 결정돼야 로드 가능 → 통신 지연이 latency를 지배.
  - 완화책: LRU 캐시, 라우팅 예측 기반 prefetch, batch 재사용, CPU에서 expert 연산.
  - 근거: `data/topics/moe-streaming/paper-moe-offloading-arxiv-2312.17238.txt`, `paper-dali-arxiv-2602.03495.txt`, `paper-cpu-gpu-collab-arxiv-2512.16473.txt`.

## colibrì의 구현 (코드 근거)
분석 대상: `external/colibri/c/glm.c`의 `moe()` (`:1270`)와 expert I/O 서브시스템.

### 1) 라우팅 (FASE A)
- 각 position에서 router 로짓 → sigmoid + bias → top-K 선택(DeepSeek-V3식 noaux_tc, `routed_scale`).
- `--topp`(adaptive expert top-p)로 실제 사용하는 expert 수 `keff`를 줄여 디스크 읽기 감소.
- 사용 통계(`eusage`, `eheat`)를 누적 → 나중에 hot-expert pin 학습에 사용.
- 코드: `glm.c:1277`~`1302`.

### 2) Batch-union (FASE B)
- S>1(prefill, MTP 검증)에서 **배치 내 unique expert를 한 번만 읽어** 그 expert로 라우팅된 모든 position에 적용.
- 가중치를 position마다 다시 읽지 않음 → I/O 절약.
- 코드: `glm.c:1318`~`1324`.

### 3) 해소: pin → cache → disk (FASE C/D)
- unique expert를 64개 블록 단위로 처리하며, 각 expert를 순서대로 조회:
  1. **pin**(hot-store)에 있으면 hit — `glm.c:1332`.
  2. 없으면 **레이어별 LRU 캐시** 조회, hit 시 `used=++eclock` 갱신 — `glm.c:1334`.
  3. 둘 다 miss면 디스크 로드 대상(`ws[]` 슬랩) — `glm.c:1336`.

### 4) 디스크 로드 (`expert_load`, `:897`)
- expert 1개 = gate/up/down 3개 텐서. 파일 오프셋 정렬 후, **연속이면 1회 `pread`**, 아니면 3회.
- **O_DIRECT** 경로(4K 정렬)와 버퍼드 fallback 모두 지원 — `glm.c:937`~`950`.
- `g_drop`이면 `posix_fadvise(...DONTNEED)`로 page cache 압박을 피함 — `glm.c:962`~`966`.

### 5) load ‖ matmul 오버랩 (PIPE)
- `g_pipe`일 때 I/O 워커 pool이 miss를 비동기로 읽고(`pipe_dispatch`, `:1054`), 메인 스레드는 필요한 expert만 `pipe_wait`(`:1064`)로 기다리며 matmul 수행.
- lock-free, **generation-tagged cursor**로 세대 간 안전성 보장(오래된 워커가 잘못된 배치 상태를 읽지 못함) — `glm.c:986`~`996` 주석.
- 코드: `glm.c:1338`~`1349`.

### 6) 비동기 readahead / prefetch
- 현재 블록 계산 중 **다음 64개 블록**을 `expert_prefetch`(`WILLNEED`)로 선반입 — `glm.c:1350`~`1361`.
- **router-lookahead(`PILOT=1`)**: 다음 레이어 라우팅을 현재 레이어 post-attention 상태로 71.6% 예측 → 전용 I/O 스레드가 선반입. `la_predict`(`:1419`), `pilot_prefetch`(`:1460`).

### 7) LRU 승격 (블록 끝)
- 이번 블록에서 로드한 miss expert를 캐시에 승격. 빈 슬롯 있으면 채우고, 없으면 `used`가 가장 오래된 슬롯을 교체(스왑 버퍼) — `glm.c:1388`~`1394`.

### 8) Hot-expert pin & 학습 캐시
- `.coli_usage`에 라우팅 빈도를 누적, 시작 시 가장 뜨거운 expert를 여유 RAM에 pin(`pin_load`, `:2409`; `pin_wire`, `:2392`).
- 라이브 재핀(`--repin N`): 세션 heat map으로 cold pin을 hot streamed expert로 교체, 25% hysteresis + 4-swap 한도로 thrashing 방지(`repin_pass`, `:2033`).

### 9) RAM 안전
- 시작 시 `MemAvailable` 기반으로 expert 캐시 크기를 자동 산정(작업셋+KV+MTP+재구성 버퍼 투영)해 OOM-killer 방지 — `mem_available_gb`(`:2505`), `expert_avail`(`:2538`). 근거: `README.md:41`, `:320`.

## colibrì가 선행 연구와 다른 점
- 대부분의 offloading은 **RAM↔GPU(PCIe)** 전송에 초점. colibrì는 **SSD↔RAM(CPU 추론)** 을 1차 경로로 삼는다.
  - 이유: expert를 매번 GPU로 올리면 디스크 병목을 PCIe 병목으로 바꿀 뿐 — `README.md:239`.
- int4 양자화로 expert당 ~19MB까지 줄여 SSD 스트리밍을 현실화(전송량이 관건).

## 한계 및 트레이드오프
- **cold decode는 디스크 바운드**: 토큰당 ~11GB 읽기 → 느린 드라이브에서 0.05–0.1 tok/s. 근거: `README.md:52`.
- 작은 RAM에서는 캐시 슬롯이 2개/레이어로 제한 → 캐시 hit이 낮아 디스크가 계속 병목. **RAM cap이 실질 제약**.
- prefetch는 디스크가 이미 포화면 이득이 적음(개발 머신에서 중립).
- SSD는 읽기 위주라 마모는 크지 않으나, 지속 발열과 (RAM 부족 시) swap write에 주의.

## 출처
- 코드: `external/colibri/c/glm.c`, `external/colibri/README.md`
- 논문: `data/topics/moe-streaming/` (SOURCE.md 참조)
