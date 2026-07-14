# 84 · (1) ThinkFlow gpt-oss-120b 스왑 운영자 런북/체크리스트

`docs/81`의 설계를 **운영자가 그대로 따라 실행**할 수 있는 체크리스트로 확정한다. 실행은 사용자가 직접(단일 H100이라 컷오버는 유지보수 창 필요). 스크립트: `scripts/thinkflow_swap_rehearsal.sh`.

## 0. 현재 상태 근거 (사용자 제공)
- GPU 80GB 중 **99.4% 예약**(20b vLLM의 KV 과다예약) · **사용률 0%**(idle) · 43°C/79W.
- CPU 97% idle · RAM 228GB 가용 · Swap 0.
- 함의: **120b를 올리려면 20b 컨테이너를 내려 VRAM을 비워야 한다**(동시 상주 불가). 컷오버는 짧은 창이면 충분(연산 idle이라 트래픽 적은 시간 선택 용이).

## 1. 사전 준비 (무중단 · 언제든 가능)
- [ ] 저장소 확보: `git clone https://github.com/atozwizard/Colibri_Survey && cd Colibri_Survey`
- [ ] `bash scripts/thinkflow_swap_rehearsal.sh --preflight` 실행하여 아래 자동 점검 통과:
  - [ ] hf_cache 디스크 여유 ≥ 70GB (서버 3.2TB → 여유)
  - [ ] `vllm/vllm-openai:v0.21.0` 이미지 존재
  - [ ] **120b 가중치 사전 다운로드**(`openai/gpt-oss-120b`, ~63GB) — 네트워크만, GPU/서빙 무관
  - [ ] VRAM 산술 sanity (util 0.92 → 73.6GB 예약, 가중치 ~63GB, KV 여유 ~10GB)
  - [ ] 롤백 자산(`/home/ubuntu/vllm/run_vllm.sh`, 20b) 존재
- [ ] **BGE를 CPU로 이전** 결정 시: `THINKFLOW_EMBED_MODEL`/`RERANKER` 디바이스를 CPU로(경로 동일). 48코어면 RAG 임베딩/리랭크 throughput 충분.
- [ ] KPI A/B 기준선 확보: 현행 20b로 `kpi` 케이스 돌려 `hit_rate`/`consistency` 스냅샷 저장.

## 2. 컷오버 (유지보수 창 · 파괴적 · CONFIRM=yes)
> 트래픽 최저 시간대 선택. 예상 다운타임 = 120b 로드 시간(수십 초~수 분).

- [ ] 공지/점검 배너(운영콘솔).
- [ ] `CONFIRM=yes bash scripts/thinkflow_swap_rehearsal.sh --cutover`
  - 20b 컨테이너 정지 → VRAM 해제
  - 120b 기동(`--served-model-name gpt-oss-20b` 유지 시 앱 config 무변경 과도기)
  - `/v1/models` 워밍업 대기 → 스모크(`/v1/chat/completions`)
- [ ] 스모크 응답 정상 확인(한국어 질의 1건).
- [ ] `nvidia-smi`로 VRAM 여유·OOM 없음 확인(32k 컨텍스트 요청 1건으로 KV 압박 테스트).

## 3. 품질 검증 (전환 확정 전)
- [ ] KPI 하네스로 120b `hit_rate`/`consistency` 측정 → 1의 20b 기준선과 **A/B 비교**.
- [ ] 판정:
  - 품질 **≥ 기존** → 전환 확정(4단계).
  - 품질 **열세/오류** → 즉시 롤백(5단계).
- [ ] RAG 지연(BGE CPU 이전 시 임베딩/리랭크 p95) 허용범위 확인.

## 4. 전환 확정
- [ ] LLM 엔드포인트(config 단일 출처) 정식 반영. 원하면 `--served-model-name`을 `gpt-oss-120b`로 개명.
- [ ] 점검 배너 해제, 모니터링 정상 확인.
- [ ] 구 컨테이너/리소스 드레인.

## 5. 롤백 (회귀 시)
- [ ] `CONFIRM=yes bash scripts/thinkflow_swap_rehearsal.sh --rollback` (20b 복귀)
- [ ] 스모크 + KPI 재확인.

## 6. 사후
- [ ] 24~48h 모니터링(지연/에러율/hit_rate 추이).
- [ ] VRAM 과다예약 재점검: 120b에 맞춰 `--gpu-memory-utilization` 재튜닝(현재 20b는 99.4% 예약 = 과다).
- [ ] 결과를 `docs/81`에 실측치로 추기(예상 대비 KV 여유·품질 델타).

## 대안 (스왑 없이도 개선)
- 스왑이 부담이면, 최소 조치로 **현행 20b의 `--gpu-memory-utilization`을 낮춰**(99.4%→적정) VRAM 여유를 확보하는 것만으로도 안정성↑. 단 품질 상한(20B)은 그대로 → 품질 목표가 크면 120b 스왑 권장.

## 참조
- 설계: `docs/81-thinkflow-upgrade-design.md`
- 스크립트: `scripts/thinkflow_swap_rehearsal.sh`
- 모델 근거: `docs/80-olmoe-and-h100-recommendations.md`
