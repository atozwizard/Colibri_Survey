# 10 · colibri 코드 구조 분석

분석 대상: `external/colibri/` (JustVugg/colibri @ `5254470`, 2026-07-13, Apache-2.0)

## 요약 (3줄)
- 런타임은 **단일 C 파일 `c/glm.c`(~2,400줄) + 소규모 헤더**로 구성되며, BLAS·Python·GPU가 필수가 아니다.
- 추론 파이프라인은 `embed → (레이어별) attention → moe/dense_mlp → lm_head`이고, MoE 레이어에서 expert를 디스크에서 스트리밍한다.
- Python·셸 도구(변환·벤치·서버)는 런타임과 분리되어 있으며, 엔진의 실행 의존성이 아니다.

## 1. 저장소 레이아웃
```text
external/colibri/
├── Makefile                 # 루트 빌드 진입점 (c/로 위임)
├── c/
│   ├── glm.c                # 단일 파일 GLM 엔진 (핵심)
│   ├── st.h                 # safetensors 로더
│   ├── tok.h, tok_unicode.h # 바이트 레벨 BPE 토크나이저 (C 구현)
│   ├── json.h               # 최소 JSON 파서 (config)
│   ├── grammar.h            # GBNF 문법 강제 draft
│   ├── compat.h             # Windows(_WIN32) POSIX 호환 계층
│   ├── backend_cuda.{cu,h}  # 선택적 CUDA 백엔드 (resident tensor)
│   ├── olmoe.c              # 소형 OLMoE 엔진 (검증/A-B용)
│   ├── iobench.c            # 디스크 I/O 벤치마크
│   ├── coli                 # 사용자용 CLI 런처 (원본 바이너리는 vendoring 제외)
│   ├── openai_server.py     # OpenAI 호환 HTTP 게이트웨이
│   ├── resource_plan.py     # `coli plan` (메모리 계획)
│   ├── doctor.py            # `coli doctor` (실행 전 점검)
│   ├── tools/               # FP8→int4 변환, fixture, 벤치
│   └── tests/               # 의존성 없는 C/Python 테스트
├── web/                     # 브라우저 UI (React+TS, 순수 OpenAI API 클라이언트)
└── desktop/                 # Tauri 데스크톱 셸
```
근거: `external/colibri/README.md:426`(Repo layout).

## 2. glm.c 내부 구조 (함수 지도)

### 2.1 양자화 / 커널
- `matmul` (`glm.c:204`), `matmul_q` (`:210`), `matmul_qt` (`:475`) — f32 / int8 / QT(양자화 텐서) 행렬곱.
- `quantize_rows` (`:512`), `qt_alloc`/`qt_fill` (`:601`/`:608`) — per-row scale 양자화(int8/int4/int2).
- AVX2 정수 dot 커널을 shape별로 선택(측정 기반). 근거: `README.md:32`.

### 2.2 모델 로딩
- `load_cfg` (`:652`) — config.json 파싱(레이어 수, head, kv_lora 등).
- `qt_from_disk`/`qt_load` (`:708`/`:722`) — safetensors에서 양자화 텐서 로드.
- `model_init` (`:738`) — dense 텐서를 RAM에 상주시키고 expert 캐시를 RAM에서 auto-size.

### 2.3 순전파 파이프라인
- `embed_row` (`:880`) — 토큰 임베딩.
- `attention` (`:1113`) — **MLA attention**(q/kv-LoRA, 부분 RoPE), 압축 KV, DSA sparse, weight absorption. → [21 문서](./21-mla-kv-compression.md)
- `moe` (`:1270`) — **MoE 라우팅 + expert 스트리밍/캐시 + batch-union**. → [20 문서](./20-moe-streaming.md)
- `dense_mlp` (`:1407`) — 초기 3개 dense 레이어용 MLP.
- `layer_forward` (`:1489`) / `layers_forward` (`:1503`) — 레이어 스택 실행.

### 2.4 Expert 스트리밍 서브시스템
- `expert_load` (`:897`) — expert 3개 텐서(gate/up/down)를 `pread`로 디스크에서 읽음. O_DIRECT/버퍼드/`posix_fadvise(DONTNEED)` 처리.
- `pipe_init`/`pipe_dispatch`/`pipe_wait` (`:1043`/`:1054`/`:1064`) — I/O 워커 pool로 load ‖ matmul 오버랩(lock-free, generation-tagged cursor).
- `expert_prefetch` (`:1070`) — `WILLNEED` 비동기 readahead.
- `pilot_prefetch` (`:1460`) / `la_predict` (`:1419`) — router-lookahead 기반 다음 레이어 expert 선반입.

### 2.5 Speculative Decoding (MTP)
- `mtp_draft` (`:1589`) — GLM-5.2 native MTP head(레이어 78)로 draft 토큰 생성.
- `mtp_absorb` (`:1627`) — 검증된 토큰을 MTP head KV에 흡수.
- `ngram_draft` (`:1570`), `grammar_draft` (`:1699`) — n-gram / 문법 강제 draft 소스.
  → [22 문서](./22-speculative-decoding.md)

### 2.6 KV 관리 · 실행 모드 · 서버
- `kv_alloc`/`kv_bind` (`:1516`/`:1533`) — 압축 KV 버퍼 할당/바인딩.
- `kv_disk_append`/`kv_disk_load` (`:2095`/`:2118`) — `.coli_kv` 지속화(대화 warm reopen).
- `run_text` (`:1951`), `generate` (`:1910`), `run_serve` (`:2173`) — 배치/생성/서버 루프.
- `pin_load`/`pin_wire` (`:2409`/`:2392`), `repin_pass` (`:2033`) — hot expert pin·라이브 재핀.
- `mem_available_gb` (`:2505`), `expert_avail` (`:2538`) — RAM 예산 기반 캐시 안전 상한(OOM 방지).

## 3. 도구 계층 (런타임 비의존)
- **변환**: `c/tools/convert_fp8_to_int4.py` — FP8 체크포인트를 샤드 단위로 다운로드→dequant(128×128 block scale)→int4 재양자화→샤드 삭제. 756GB를 한 번에 둘 필요 없음. 근거: `README.md:42`.
- **계획**: `coli plan` — safetensors 헤더만 읽어 dense/expert footprint, RAM reserve, expert-cache cap, VRAM hot tier를 JSON으로 출력. 근거: `README.md:107`.
- **점검**: `coli doctor` — 모델 디렉토리·config·토크나이저·헤더·RAM·CUDA 링크를 read-only로 검증. 근거: `README.md:116`.
- **서버**: `coli serve` — 표준 라이브러리만으로 OpenAI 호환 HTTP. 단일 모델 프로세스 + FIFO 큐, `--kv-slots`로 최대 16개 독립 KV 컨텍스트. 근거: `README.md:171`, `:218`.

## 4. 플랫폼 이식
- 모든 플랫폼 차이는 `c/compat.h`에 격리(POSIX I/O → Windows API: `pread`→`ReadFile+OVERLAPPED` 등). 엔진 소스는 불변.
- 근거: `README.md:133`.

## 한계 및 관찰
- 정수 커널이 shape 의존적이라, batched(S>1)/GPU 경로가 single-token 경로와 미세하게 다르게 반올림 → int4 argmax tie가 뒤집혀 **동일 프롬프트에서 스트림이 byte-identical하지 않을 수 있음**. 근거: `README.md:29`.
  - byte-exact 재현: `DRAFT=0`(no spec) + `IDOT=0 COLI_CUDA=0`(kernel/GPU 독립).
- 단일 파일 설계는 가독성 트레이드오프가 있으나, 런타임 의존성 최소화라는 목표에 부합.

## 출처
- 코드: `external/colibri/c/glm.c`, `external/colibri/README.md`
- vendoring 출처/커밋: `../external/README.md`
