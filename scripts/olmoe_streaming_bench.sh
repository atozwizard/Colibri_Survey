#!/usr/bin/env bash
# (b) OLMoE 실가중치 스트리밍 실측 — ThinkFlow H100 박스 '공유 안전' 버전.
#
# 핵심: colibri 스트리밍 벤치는 CPU + 디스크 I/O + RAM 만 사용한다(GPU 무관).
# 따라서 ThinkFlow의 GPU LLM 서빙에는 영향이 없다. 경쟁 자원은 CPU/RAM/디스크뿐이며,
# 아래 격리(cgroup/nice/ionice/taskset) + 소형 모델(OLMoE 1B활성) + 오프피크로 최소화한다.
#
# 사용:
#   BASE=/home/ubuntu/bench CORES=0-3 MEM=8G bash scripts/olmoe_streaming_bench.sh
set -euo pipefail

BASE="${BASE:-$PWD/olmoe_bench}"
REPO="${REPO:-allenai/OLMoE-1B-7B-0924-Instruct}"
C_DIR="${C_DIR:-external/colibri/c}"
OUT_I4="$BASE/olmoe_i4"
CORES="${CORES:-0-3}"          # taskset: 소수 코어에만 고정(ThinkFlow CPU 보호)
MEM="${MEM:-8G}"               # MemoryMax: RAM 상한
IOW="${IOW:-10}"               # IOWeight: 디스크 우선순위 낮춤(기본 100)
mkdir -p "$BASE"

# 자원 격리 래퍼: cgroup v2(systemd-run) 우선, 없으면 nice+ionice+taskset.
guard(){
  if command -v systemd-run >/dev/null 2>&1; then
    systemd-run --scope --quiet \
      -p "CPUQuota=$(( $(nproc) > 8 ? 400 : 200 ))%" \
      -p "MemoryMax=$MEM" -p "IOWeight=$IOW" \
      nice -n 19 ionice -c3 taskset -c "$CORES" "$@"
  else
    nice -n 19 ionice -c3 taskset -c "$CORES" "$@"
  fi
}

echo "== 0) 엔진 + iobench 빌드 =="
make -C "$C_DIR" olmoe iobench

echo "== 1) 디스크 특성화 (iobench, 초경량·GPU무관) =="
# 스트리밍 속도의 물리 상한 = 이 저장소의 랜덤 read 대역폭.
# 모델 파일이 아직 없으면 스크래치 파일로 측정.
PROBE="$OUT_I4/../_iobench_probe.bin"
[ -f "$PROBE" ] || { echo "  (probe 4GB 생성)"; head -c 4G /dev/zero > "$PROBE" 2>/dev/null || dd if=/dev/zero of="$PROBE" bs=1M count=4096 status=none; }
echo "  ./iobench <file> <blockMB=1> <reads=2000> <threads=4> <direct=1>"
guard "$C_DIR/iobench" "$PROBE" 1 2000 4 1 || true

echo "== 2) OLMoE 변환(int8 expert) — 최초 1회 =="
if [ ! -f "$OUT_I4/config.json" ]; then
  guard uv run --python 3.12 --with torch --with safetensors --with huggingface_hub \
    python "$C_DIR/tools/convert_olmoe.py" --repo "$REPO" --out "$OUT_I4"
fi
df -h "$OUT_I4" | tail -1

echo "== 3) cap 스윕 (격리 실행) — 캐시 축소로 스트리밍 강제 =="
# OLMoE: 64 experts/layer, top-8. cap<64면 축출 발생.
# 엔진: SNAP=<dir> ./olmoe <cap> <expert_bits>
printf "%-6s %s\n" "cap" "engine-metrics(hit rate / tok-s / RSS / expert-disk)"
for cap in 64 32 16 8 4; do
  raw=$(SNAP="$OUT_I4" guard "$C_DIR/olmoe" "$cap" 8 2>/dev/null || true)
  metr=$(echo "$raw" | grep -oiE "hit rate [0-9.]+%|[0-9.]+ tok/s|RSS [0-9.]+ GB|expert-disk [0-9.]+s" | tr '\n' ' ')
  printf "%-6s %s\n" "$cap" "${metr:-(원시 로그 확인)}"
done

echo
echo "== 완료 =="
echo "해석: cap↓ → hit rate↓ → expert-disk 시간↑ → tok/s↓  (메모리↔속도 곡선)."
echo "동시에 ThinkFlow QPS/지연을 로깅해 상관 배제. drop_caches 는 공유박스에서 금지."
