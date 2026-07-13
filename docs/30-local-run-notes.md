# 30 · 로컬 실행/빌드 노트 (Apple Silicon)

이 서베이를 진행한 로컬 환경에서 colibri 엔진을 실제 빌드하고 `coli doctor`로 준비 상태를 점검한 기록.

## 요약 (3줄)
- 이 머신은 **Apple Silicon(arm64) + Apple clang + Homebrew libomp** 환경이다.
- colibri Makefile이 macOS/arm64를 지원하므로 **AVX2 없이 NEON/스칼라 경로 + libomp 멀티스레드**로 빌드에 성공했다.
- 모델 가중치(~370GB)는 받지 않았으나, `coli doctor`가 정상 실행되어 **엔진 바이너리 ready**를 확인했다.

## 환경
| 항목 | 값 |
|---|---|
| CPU/arch | Apple Silicon (aarch64) |
| 컴파일러 | Apple clang 21.0.0 (진짜 gcc 아님) |
| OpenMP | Homebrew `libomp` @ `/opt/homebrew/opt/libomp` (설치됨 → 멀티스레드) |
| Python | 3.14.5 |
| 패키지 관리 | **uv 0.11.16** |
| AVX2 | 없음(arm64) → colibri의 정수 AVX2 커널은 NEON/스칼라 fallback |

## 의존성: uv
런타임 C 엔진과 `coli {doctor,plan,build,info}` CLI는 **순수 표준 라이브러리**라 추가 파이썬 의존성이 없다.
루트 [`pyproject.toml`](../pyproject.toml)에 선택적 의존성만 정의했다.

```bash
# 가상환경 생성 (Python 3.14)
uv venv --python 3.14

# CLI는 의존성 없이 바로 실행 가능
uv run --no-sync python external/colibri/c/coli --help

# 무거운 도구는 필요할 때만:
uv sync --extra convert   # coli convert (torch, safetensors, huggingface_hub, numpy)
uv sync --extra bench     # coli bench  (tokenizers, datasets)
```

## 빌드
```bash
make -C external/colibri/c glm
# 실제 실행된 컴파일 라인:
# clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include ... glm.c -o glm -lm -L.../libomp/lib -lomp
```
- 산출물: `external/colibri/c/glm` (약 207KB). **바이너리이므로 커밋 대상 아님**(.gitignore 처리).

## coli doctor 결과 (모델 미보유)
```text
colibri doctor · .../glm52_i4
[fail] model.path         model directory does not exist
[fail] model.config       config.json is missing or invalid
[fail] model.tokenizer    tokenizer.json is missing
[skip] storage.persistence persistence requires a model directory
[  ok] engine.binary      engine executable is ready      ← 빌드 검증됨
[skip] accelerator.cuda   no NVIDIA GPU detected; CPU path is available
[fail] model.shards       missing config.json
[skip] storage.disk       storage check requires a valid model
[skip] memory.ram         RAM projection requires a valid model
[skip] placement.plan     placement requires a valid model
result error
```
- `engine.binary … ready`, `accelerator.cuda … CPU path available`는 기대대로 통과.
- `model.*`는 가중치를 받지 않았으므로 fail(정상). 실제 추론까지 하려면
  `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`(~370GB)를 NVMe에 받고 `COLI_MODEL`을 지정해야 한다.

## 다음 단계(실제 추론 시)
```bash
COLI_MODEL=/nvme/glm52_i4 uv run --no-sync python external/colibri/c/coli doctor   # 준비 재점검
COLI_MODEL=/nvme/glm52_i4 uv run --no-sync python external/colibri/c/coli plan      # 자원 계획
COLI_MODEL=/nvme/glm52_i4 uv run --no-sync python external/colibri/c/coli chat      # 대화
```
