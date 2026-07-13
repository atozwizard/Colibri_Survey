# 22 · Speculative Decoding (MTP 기반)

## 요약 (3줄)
- Speculative decoding은 가벼운 draft로 여러 토큰을 **미리 제안**하고, target 모델이 **한 번의 배치 forward로 검증**해 맞는 prefix를 받아들여 step 수를 줄인다.
- **MTP(Multi-Token Prediction)** 는 별도 draft 모델 없이, 모델 자체의 multi-token 예측 head를 draft로 쓰는 self-speculative 방식이다(Medusa/EAGLE 계열과 유사).
- colibrì는 GLM-5.2의 **native MTP head(레이어 78)** 로 draft를 만들고 검증한다. **head는 int8이어야** acceptance가 살아난다(int4면 0–4%).

## 배경 / 문제의식
- 자기회귀 디코딩은 토큰마다 1 forward → 매번 파라미터를 메모리에서 끌어와야 해 지연이 크다.
- Speculative decoding 3단계: (1) draft로 γ개 제안, (2) target이 병렬 검증, (3) 일치 prefix 수락 + 불일치 시 fallback.
  - draft-model 방식(별도 소형 모델) / **self-speculative**(Medusa 다중 head, MTP, EAGLE)로 나뉨.
  - 검증 전략: **rejection sampling**(분포 보존, 무손실) vs **typical acceptance**(휴리스틱, 비탐욕 시 손실 가능).
  - 근거: `data/topics/speculative-decoding/paper-medusa.txt`, `primer-amanai-speculative-decoding.txt`.
- GLM-5.2는 MTP 레이어를 speculative decoding용으로 개선해 accepted length를 최대 20% 향상, rejection sampling 도입. 근거: `data/glm/notes.md`.

## colibrì의 구현 (코드 근거)
분석 대상: `external/colibri/c/glm.c`.

### 1) MTP draft 생성 (`mtp_draft`, `:1589`)
- DeepSeek-V3식 체인: `h' = Layer78( eh_proj[ enorm(emb(tok)) ; hnorm(h) ] )`, `draft = argmax(lm_head(mtp_norm(h')))`.
- 직전 hidden `hlast`에서 시작해 G개 draft를 순차 생성(각 draft를 다음 입력으로 사용).
- MTP head의 KV는 레이어 `n_layers` 행에 존재하며 decode 전용 윈도우(prefill 불필요).
- 코드: `glm.c:1595`~`1620`.

### 2) draft 검증 & 흡수
- 생성 루프(`run_text`/`generate`)에서 draft를 **batch-union forward**로 한 번에 검증 → 일치 prefix 수락.
- 검증된 (emb(token@pos+1), h_true@pos) 쌍을 MTP head KV에 흡수: `mtp_absorb` (`:1627`). 배치 1회 layer_forward로 처리(batch-union 덕에 expert 저렴).

### 3) 추가 draft 소스 (합성)
- **n-gram draft** (`ngram_draft`, `:1570`): 최근 bigram 재등장 위치의 후속 토큰을 draft로.
- **문법 강제 draft** (`grammar_draft`, `:1699`): GBNF가 유일 합법 바이트만 허용하는 span(중괄호/따옴표/키명/enum)을 pre-accepted draft로 주입(≈1.0 acceptance). JSON/함수호출 등 제약 출력에서 강력. int4 MTP head에서도 동작. `GRAMMAR=file.gbnf`, `GRAMMAR_DRAFT=n`.
- 세 소스(MTP / n-gram / grammar)는 같은 batch-union forward에서 함께 검증 → 잘못된 draft는 rejected일 뿐 출력에 영향 없음.

### 4) int8 head의 중요성
- int4 MTP head → draft acceptance 0–4%로 붕괴(speculation 미작동). int8 → **39–59% acceptance, 2.2–2.8 tok/forward**.
- 모델 다운로드 시 `out-mtp-*`가 int8인지 확인 필요. 근거: `README.md:29`,`:67`.

## 정확성(무손실) 관련 주의
- **정확 산술에서는 무손실**이나, colibrì의 정수 커널은 shape 의존적이라 batched(S>1)/GPU forward가 single-token 경로와 미세하게 다르게 반올림.
- int4 GLM-5.2는 argmax tie에 가까워, 이 반올림 차이가 토큰을 뒤집어 **탐욕 출력이 non-speculative와 byte-identical하지 않을 수 있음**(단, 매 토큰은 여전히 valid forward의 argmax → 연속은 정상).
- byte-exact: `DRAFT=0`(+ `IDOT=0 COLI_CUDA=0`). 샘플링 시 rejection sampling이 분포 보존. 근거: `README.md:29`.
- **cold cache 주의**: 검증되는 draft가 추가 expert를 라우팅(~660→~1100 expert-load/token)하므로, 캐시/pin이 데워지기 전엔 speculation이 순 time loss일 수 있음.

## 한계 및 트레이드오프
- 이득은 acceptance rate에 비례. head 정밀도(int8)와 캐시 온도가 실효 속도를 좌우.
- 50% 미만 acceptance면 adaptive guard가 문법 소스를 끔.

## 출처
- 코드: `external/colibri/c/glm.c:1570`~`1646`, `README.md:29`,`:67`
- 자료: `data/topics/speculative-decoding/`, `data/glm/notes.md`
