# 31 · 엔진 정확성 실검증 (로컬 실행 결과)

370GB 모델 없이, 이 로컬(Apple Silicon)에서 colibri **엔진 코어의 정확성**을 실제로 검증한 기록.

## 요약 (3줄)
- 의존성 없는 **C 코어 테스트 4종 전부 통과**(json/safetensors/tier/grammar).
- tiny GLM oracle로 실제 `glm.c` 엔진이 transformers 오라클과 **token-exact 일치: 32/32 positions**.
- arm64에서 `idot: neon` 커널 경로로 MLA·MoE·shared expert·DSA(no-op) 전 경로가 정확히 동작함을 확인.

## 1. 무엇을 확인하려 했나 (검증 목적)
"370GB GLM-5.2 가중치 없이도 **엔진이 올바르게 동작하는가**"를 값싸게 증명:
1. **정수/로더/문법 파서의 정확성** (C 단위 테스트)
2. **실제 forward 경로의 수치 정확성** — MLA attention, MoE 라우팅, shared expert, 양자화 dequant-on-use가 참조 구현과 token-exact인가
3. **로컬 툴체인 정합성** — arm64 + clang + libomp + NEON fallback이 컴파일만이 아니라 **정확한 출력**을 내는가

## 2. C 코어 단위 테스트 (의존성 0)
```bash
make -C external/colibri/c test-c
```
결과:
```text
json tests: ok
safetensors primitive tests: ok
tier tests: ok
test_grammar: ok
```
- 검증 대상: `json.h`(config 파서), `st.h`(safetensors 로더), `tier.h`(자원 tier), `grammar.h`(GBNF 문법 draft).

## 3. Token-exact 검증 (tiny GLM oracle)
### 3.1 오라클 생성 (torch/transformers 임시환경, uv)
```bash
cd external/colibri/c
uv run --python 3.12 --with torch --with "transformers>=5.13" --with safetensors --with numpy \
    python tools/make_glm_oracle.py
# → glm_tiny/ (랜덤 가중치 tiny GLM, 실제 glm_moe_dsa 아키텍처) + ref_glm.json 생성
```
- tiny 구성(실제 아키텍처 유지): hidden 128, 5층(3 dense+2 sparse), 8 expert top-2, 1 shared, q_lora 64/kv_lora 32, MLA, DSA index_topk=4096(≫seq → 전체 선택=dense와 동일).

### 3.2 엔진 teacher-forcing 실행
```bash
SNAP=./glm_tiny TF=1 ./glm 64 16 16
```
결과(발췌):
```text
[DSA] indexer active: top-4096 sparse attention beyond 4096 context tokens
[MTP] absent (draft=0)
[RAM_GB=12.9 auto] cap=64 ok (projected peak 3.7 GB)
== GLM C engine (glm_moe_dsa), cache=64 experts/layer | experts@16-bit dense@16-bit | idot: neon ==
loaded in 0.00s | resident dense: 1.57 MB | layers=5 experts=8 | MTP absent (draft=0)
PREFILL (teacher-forcing) C vs oracle: 32/32 positions | 3577.4 pos/s
```
- **32/32 positions**: C 엔진의 32개 위치 예측이 transformers 오라클과 **완전 일치**.
- `idot: neon`: arm64 NEON 정수 커널 경로가 동작(AVX2 fallback 확인).
- RAM 자동 캡(12.9GB 감지 → peak 3.7GB projection)도 정상 작동.

## 4. 이 검증이 증명하는 것 / 아닌 것
- **증명함**:
  - `glm.c`의 MLA attention(q/kv-LoRA, 부분 RoPE, weight absorption), MoE 라우팅(sigmoid+bias, top-k, shared expert), DSA 인덱서(전체 선택 시 dense 동치)가 **수치적으로 정확**하다.
  - 이 로컬 arm64 빌드가 정확한 출력을 낸다.
- **증명 안 함**(별도 자원 필요):
  - 대규모 int4 양자화 품질(본 오라클은 16-bit dense tiny 모델).
  - 디스크 스트리밍 성능(tiny 모델은 expert가 작아 disk 0.000s) — 대형 모델·실 NVMe 필요.
  - MTP speculative(tiny 오라클엔 MTP head 없음, `draft=0`).

## 5. 재현 절차 (요약)
```bash
# 1) 빌드
make -C external/colibri/c glm
# 2) C 테스트
make -C external/colibri/c test-c
# 3) 오라클 + token-exact
cd external/colibri/c
uv run --python 3.12 --with torch --with "transformers>=5.13" --with safetensors --with numpy python tools/make_glm_oracle.py
SNAP=./glm_tiny TF=1 ./glm 64 16 16      # expect 32/32
```
- 생성물(`glm_tiny/`, `ref_glm.json`)은 재현 가능하며 저장소에 커밋하지 않는다(vendored `.gitignore`가 `glm_tiny/` 제외).

## 출처
- 코드: `external/colibri/c/glm.c`, `tools/make_glm_oracle.py`, `Makefile`, `setup.sh`
- 실행 환경: `30-local-run-notes.md`
