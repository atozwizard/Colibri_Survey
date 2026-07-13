#!/usr/bin/env bash
# (b) OLMoE 실가중치 스트리밍 실측 — ThinkFlow H100 박스(x86 Linux) 권장 실행.
#
# colibri의 디스크 스트리밍/LRU 성능을 실모델(OLMoE-1B-7B)로 측정한다.
# 캐시 용량(cap)을 낮춰 강제로 miss를 유발하고 hit rate·expert-disk 시간·tok/s를 기록.
#
# 전제:
#   - colibri C 엔진 빌드 완료(make -C external/colibri/c olmoe 또는 glm)
#   - 모델은 NVMe/ext4 등 빠른 로컬 저장소에 둘 것(네트워크/USB 금지)
#   - torch/safetensors/huggingface_hub 사용 가능(uv 또는 conda)
#
# 사용:
#   BASE=/home/ubuntu/bench bash scripts/olmoe_streaming_bench.sh
set -euo pipefail

BASE="${BASE:-$PWD/olmoe_bench}"
REPO="${REPO:-allenai/OLMoE-1B-7B-0924-Instruct}"
C_DIR="${C_DIR:-external/colibri/c}"
OUT_I4="$BASE/olmoe_i4"
mkdir -p "$BASE"

echo "== 0) 엔진 빌드 =="
make -C "$C_DIR" olmoe

echo "== 1) OLMoE 변환(int4/int8 expert) =="
if [ ! -f "$OUT_I4/config.json" ]; then
  uv run --python 3.12 --with torch --with safetensors --with huggingface_hub \
    python "$C_DIR/tools/convert_olmoe.py" --repo "$REPO" --out "$OUT_I4"
fi

echo "== 2) 저장소 위치 안내 =="
df -h "$OUT_I4" | tail -1
echo "   (스트리밍 성능은 이 저장소의 랜덤 read IOPS/대역폭에 좌우됨)"

echo "== 3) cap 스윕: 캐시 용량을 줄여 스트리밍 강제 =="
# OLMoE-1B-7B: 64 experts/layer, top-8. cap<64면 축출 발생.
# olmoe 엔진 인자: SNAP=<dir> ./olmoe <cap> <expert_bits> [ref]
printf "%-6s %-14s %-14s\n" "cap" "hit_rate" "note"
for cap in 64 32 16 8 4; do
  # posix_fadvise(DONTNEED)로 페이지캐시 영향 최소화 위해 매 실행 새 프로세스
  line=$(SNAP="$OUT_I4" "$C_DIR/olmoe" "$cap" 8 2>/dev/null \
           | grep -oE "hit rate [0-9.]+%|hit [0-9.]+%|tok/s|[0-9.]+ tok/s" | tr '\n' ' ' || true)
  printf "%-6s %s\n" "$cap" "${line:-(엔진 출력 파싱 필요: 원시 로그 확인)}"
done

echo
echo "== 완료 =="
echo "해석: cap↓ → hit rate↓ → expert-disk 시간↑ → tok/s↓ 가 스트리밍의 실제 비용."
echo "권장 리포팅: (cap, hit_rate, expert-disk s, tok/s, RSS) 표 + 저장소 spec."
