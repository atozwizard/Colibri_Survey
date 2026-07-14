#!/usr/bin/env bash
# (1) ThinkFlow LLM 무중단 스왑 리허설: gpt-oss-20b -> gpt-oss-120b (H100 80GB 단일).
#
# 단일 H100은 20b/120b 동시 상주 불가 → 리허설을 두 단계로 분리:
#   --preflight : 현행 서빙 무중단. 다운로드/용량/이미지/설정만 사전 점검·준비.
#   --cutover   : 유지보수 창에서만. 20b 내림 → 120b 기동 → 스모크 → (KPI A/B) → 유지/롤백.
#   --rollback  : 20b로 즉시 복귀.
#
# 안전장치: 파괴적 동작은 --cutover 에서만, 그리고 CONFIRM=yes 필요.
set -euo pipefail

MODE="${1:---preflight}"
HF_CACHE="${HF_CACHE:-/home/ubuntu/hf_cache}"
VLLM_DIR="${VLLM_DIR:-/home/ubuntu/vllm}"
NEW_MODEL="${NEW_MODEL:-openai/gpt-oss-120b}"
OLD_RUN="${OLD_RUN:-$VLLM_DIR/run_vllm.sh}"          # 기존 20b 기동 스크립트(롤백용)
PORT="${PORT:-8080}"
UTIL="${UTIL:-0.92}"
MAXLEN="${MAXLEN:-32768}"
SERVED_NAME="${SERVED_NAME:-gpt-oss-20b}"           # 과도기: 앱 config 무변경 위해 이름 유지
NEED_GB="${NEED_GB:-70}"                             # 120b MXFP4 다운로드 여유(대략)

log(){ printf "\033[1m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }

preflight(){
  log "PREFLIGHT (무중단) — gpt-oss-120b 스왑 준비"

  log "1) 디스크 여유 (hf_cache: $HF_CACHE)"
  df -h "$HF_CACHE" | tail -1
  avail=$(df -BG --output=avail "$HF_CACHE" | tail -1 | tr -dc '0-9')
  [ "${avail:-0}" -ge "$NEED_GB" ] && log "   OK (>= ${NEED_GB}G)" || { log "   부족: ${avail}G < ${NEED_GB}G"; exit 1; }

  log "2) GPU 상태(현행 20b 서빙 확인, 방해 없음)"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader || true

  log "3) vLLM 이미지 존재"
  docker image inspect vllm/vllm-openai:v0.21.0 >/dev/null 2>&1 \
    && log "   image ok" || { log "   pull 필요"; docker pull vllm/vllm-openai:v0.21.0; }

  log "4) 신규 모델 사전 다운로드(네트워크만, GPU 무관)"
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="$HF_CACHE" huggingface-cli download "$NEW_MODEL" --exclude "*.pt" "original/*" || \
      log "   (다운로드 재시도 권장 — 네트워크 확인)"
  else
    log "   huggingface-cli 없음 → uv 로 임시 실행"
    HF_HOME="$HF_CACHE" uv run --with huggingface_hub \
      python -c "from huggingface_hub import snapshot_download as s; s('$NEW_MODEL')" || true
  fi

  log "5) VRAM 예산 sanity (120b MXFP4 ~63GB + KV, util=$UTIL, BGE는 CPU 권장)"
  awk -v u="$UTIL" 'BEGIN{printf "   reserve=%.1fGB of 80GB; weights~63GB -> KV여유~%.1fGB\n", u*80, u*80-63}'

  log "6) 롤백 자산 확인: $OLD_RUN"
  [ -f "$OLD_RUN" ] && log "   ok(20b 복귀 가능)" || log "   경고: 20b run 스크립트 없음 → 롤백 대비 필요"

  log "PREFLIGHT 완료. 컷오버는 유지보수 창에서: CONFIRM=yes bash $0 --cutover"
}

cutover(){
  [ "${CONFIRM:-no}" = "yes" ] || { log "안전: CONFIRM=yes 필요(파괴적). 중단."; exit 1; }
  log "CUTOVER — 유지보수 창 (단일 GPU: 20b 내리고 120b 올림)"

  log "1) 현행 컨테이너 정지(20b)"; docker rm -f vllm 2>/dev/null || true

  log "2) 120b 기동 :$PORT"
  docker run -d --name vllm --restart unless-stopped --gpus all --ipc host \
    -p "$PORT:8000" -v "$HF_CACHE:/root/.cache/huggingface" \
    vllm/vllm-openai:v0.21.0 \
      --model "$NEW_MODEL" --served-model-name "$SERVED_NAME" \
      --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --enable-prefix-caching

  log "3) 워밍업 대기(모델 로드)"
  for i in $(seq 1 60); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { log "   ready"; break; }
    sleep 10; [ "$i" = 60 ] && { log "   기동 실패 → 로그 확인: docker logs vllm"; exit 1; }
  done

  log "4) 스모크 테스트"
  curl -sf "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"핑\"}],\"max_tokens\":8}" \
    | head -c 400; echo

  log "5) KPI A/B: 기존 하네스로 hit_rate/consistency 비교(수동/자동)."
  log "   품질 회귀 없으면 유지, 회귀면: CONFIRM=yes bash $0 --rollback"
}

rollback(){
  log "ROLLBACK — 20b 복귀"
  docker rm -f vllm 2>/dev/null || true
  [ -f "$OLD_RUN" ] && bash "$OLD_RUN" || { log "20b run 스크립트 없음"; exit 1; }
  log "복귀 완료."
}

case "$MODE" in
  --preflight) preflight ;;
  --cutover)   cutover ;;
  --rollback)  rollback ;;
  *) echo "usage: $0 [--preflight|--cutover|--rollback]"; exit 2 ;;
esac
