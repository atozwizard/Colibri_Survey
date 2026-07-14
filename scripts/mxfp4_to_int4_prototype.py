#!/usr/bin/env python3
"""(a) gpt-oss MXFP4 -> colibri int4 변환기 프로토타입.

목적: gpt-oss(openai/gpt-oss-20b/120b)의 MoE expert 가중치가 저장된
MXFP4(OCP Microscaling FP4: E2M1 원소 + 32개마다 E8M0 공유 스케일) 포맷을
colibri C 엔진이 읽는 int4 컨테이너(row-wise nibble packing + F32 per-row scale)로
변환한다. 양자화 수학은 colibri `tools/convert_fp8_to_int4.py`의 quant_int4와 '동일'하게
맞춰, 엔진이 기존 GLM 경로와 같은 dequant-on-use로 처리하도록 한다.

검증된 레이아웃(openai/gpt-oss-20b safetensors 헤더, 2026-07 확인):
 - MXFP4는 EXPERT MLP만(config.quantization_config.modules_to_not_convert:
   attn/router/embed/lm_head 제외).
 - per-layer grouped(32 experts):
     experts.gate_up_proj_blocks  U8 [E=32, O=5760, K/32=90, 16B]   (K=2880=hidden)
     experts.gate_up_proj_scales  U8 [E=32, O=5760, 90]              (블록당 1 E8M0)
     experts.gate_up_proj_bias   BF16 [E=32, 5760]
     experts.down_proj_blocks     U8 [E=32, O=2880, K/32=90, 16B]   (K=2880=inter)
     experts.down_proj_scales     U8 [E=32, O=2880, 90]
     experts.down_proj_bias      BF16 [E=32, 2880]
   16B/block × 2 nibble/B = 32 element/block; 90×32 = 2880 = K. ✓
 - 나머지(norm/attn/router/embed/lm_head)는 bf16 → 엔진이 F32 승격, 그대로 저장.

상태: 프로토타입/설계 검증용.
 - `--selftest`: 실제 블록 레이아웃([O,nb,16])으로 MXFP4 dequant + int4 round-trip 검증.
 - `--model <dir> --out <dir>`: grouped expert를 per-expert [O,K] int4로 변환.
 - 주의: 본 변환기는 '양자화 스테이지'만 담당. gpt-oss는 glm_moe_dsa가 아님
   (GQA+sliding-window, MLA/DSA 없음) → colibri 실행엔 엔진 어댑터 필요(docs/61).

USAGE
  python scripts/mxfp4_to_int4_prototype.py --selftest
  python scripts/mxfp4_to_int4_prototype.py --model ./gpt-oss-20b --out ./gptoss_i4 --ebits 4
"""
import argparse
import sys

import numpy as np

# ---------- MXFP4 (E2M1) 디코드 테이블 ----------
# 4bit = [sign(1) | exp(2) | mantissa(1)]. 크기값 LUT(부호 별도).
E2M1_ABS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def mxfp4_nibble_to_f32(nib: np.ndarray) -> np.ndarray:
    """nib: uint8 [-> 하위 4bit 사용]. 반환: f32 (E8M0 스케일 미적용, 원소값만)."""
    nib = nib & 0x0F
    sign = np.where((nib & 0x08) != 0, -1.0, 1.0).astype(np.float32)
    mag = E2M1_ABS[(nib & 0x07)]
    return sign * mag


def e8m0_to_scale(exp_byte: np.ndarray) -> np.ndarray:
    """E8M0: 부호/가수 없는 8bit 지수. 값 = 2^(e-127). e==255는 NaN(여기선 0 처리)."""
    exp_byte = exp_byte.astype(np.int32)
    scale = np.exp2(exp_byte - 127).astype(np.float32)
    scale[exp_byte == 255] = 0.0
    return scale


def dequant_mxfp4(blocks: np.ndarray, scales: np.ndarray, block: int = 32) -> np.ndarray:
    """gpt-oss 실제 레이아웃 dequant.
       blocks: uint8 [O, nb, 16]  (블록당 16바이트 = 32 nibble, little-nibble first)
       scales: uint8 [O, nb]      (블록당 1 E8M0)
       반환:   f32   [O, nb*32=K].
    호환: blocks가 2D [O, K/2]면 nb=K/2 취급(옛 경로)도 처리."""
    if blocks.ndim == 3:
        O, nb, bpb = blocks.shape                    # bpb=16
        lo = mxfp4_nibble_to_f32(blocks & 0x0F)      # [O,nb,16]
        hi = mxfp4_nibble_to_f32(blocks >> 4)
        vals = np.empty((O, nb, bpb * 2), np.float32)
        vals[..., 0::2] = lo
        vals[..., 1::2] = hi                         # [O,nb,32]
        sc = e8m0_to_scale(scales)[..., None]        # [O,nb,1]
        vals = vals * sc
        return vals.reshape(O, nb * bpb * 2)         # [O, K]
    # ---- 폴백: 2D [.., I/2] ----
    lo = mxfp4_nibble_to_f32(blocks & 0x0F)
    hi = mxfp4_nibble_to_f32(blocks >> 4)
    vals = np.empty(blocks.shape[:-1] + (blocks.shape[-1] * 2,), dtype=np.float32)
    vals[..., 0::2] = lo
    vals[..., 1::2] = hi
    I = vals.shape[-1]
    sc = e8m0_to_scale(scales)
    sc = np.repeat(sc, block, axis=-1)[..., :I]
    return vals * sc


# ---------- colibri int4 양자화 (convert_fp8_to_int4.py quant_int4와 동일) ----------
def quant_int4(w: np.ndarray, bits: int = 4):
    """w: [O, I] f32 -> (qbytes U8 [O*ceil(I/2)], scale f32 [O]). nibble [-8,7]."""
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1                     # 7
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, : v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, : v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)


def dequant_int4(qbytes: np.ndarray, scale: np.ndarray, O: int, I: int) -> np.ndarray:
    rb = (I + 1) // 2
    q = qbytes.reshape(O, rb).astype(np.int32)
    out = np.zeros((O, I), np.float32)
    out[:, 0::2] = (q & 0x0F) - 8
    if I > 1:
        out[:, 1::2] = (q >> 4) - 8
    return out * scale[:, None]


# ---------- selftest ----------
def selftest() -> int:
    rng = np.random.default_rng(0)
    # gpt-oss down_proj 한 expert 형상 축소판: O=8, nb=4 -> K=128
    O, nb, block = 8, 4, 32
    K = nb * block
    blocks = rng.integers(0, 256, size=(O, nb, 16), dtype=np.uint8)  # [O,nb,16B]
    scales = rng.integers(118, 130, size=(O, nb), dtype=np.uint8)    # [O,nb]
    w = dequant_mxfp4(blocks, scales, block)
    assert w.shape == (O, K), w.shape
    I = K
    # int4 재양자화 후 dequant → 상대오차 확인
    qb, sc = quant_int4(w)
    wr = dequant_int4(qb, sc, O, I)
    denom = np.maximum(np.abs(w).max(), 1e-8)
    rel = np.abs(wr - w).max() / denom
    print(f"[selftest] MXFP4 dequant shape ok {w.shape}")
    print(f"[selftest] int4 round-trip max rel-err = {rel:.4f} (기대: <~0.15, 4bit 재양자화)")
    print("[selftest] packing 규약: nibble=[-8,7]+8, little-nibble first (glm.c와 동일)")
    ok = rel < 0.25
    print("[selftest]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------- 실제 변환(스켈레톤) ----------
def convert_checkpoint(model_dir: str, out_dir: str, ebits: int) -> int:
    """gpt-oss HF 체크포인트 -> colibri int4 스냅샷.
    grouped expert 텐서 [E,O,nb,16] 를 per-expert [O,K] int4로 변환."""
    try:
        from pathlib import Path

        from safetensors.numpy import load_file, save_file
    except ImportError as exc:
        sys.exit(f"safetensors 필요: {exc} (uv run --with safetensors ...)")
    import json
    import re
    import shutil

    src = Path(model_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (src / "config.json").is_file():
        shutil.copy2(src / "config.json", out / "config.json")

    # per-layer grouped: ...experts.{gate_up_proj|down_proj}_blocks : [E,O,nb,16]
    BLOCK_RE = re.compile(r"experts\.(gate_up_proj|down_proj)_blocks$")
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"safetensors 없음: {src}")

    manifest = {"experts": [], "layout": "gpt-oss grouped [E,O,nb,16] -> per-expert [O,K] int4"}
    for si, shard in enumerate(shards, 1):
        t = load_file(str(shard))
        out_t = {}
        for name in list(t.keys()):
            if BLOCK_RE.search(name):
                base = name[: -len("_blocks")]           # ...experts.gate_up_proj
                scales = t.get(base + "_scales")
                if scales is None:
                    sys.exit(f"scales 없음: {base}_scales")
                blocks = t[name].astype(np.uint8)        # [E,O,nb,16]
                scales = scales.astype(np.uint8)         # [E,O,nb]
                E = blocks.shape[0]
                for e in range(E):                       # expert 분해
                    w = dequant_mxfp4(blocks[e], scales[e])   # [O,K]
                    qb, sc = quant_int4(w, ebits)
                    ekey = base.replace("experts.", f"experts.{e}.")
                    out_t[ekey] = qb
                    out_t[ekey + ".qs"] = sc
                    if e == 0:
                        manifest["experts"].append(
                            {"name": base, "E": int(E), "O": int(w.shape[0]), "K": int(w.shape[1])})
            elif name.endswith("_scales"):
                continue                                 # blocks에서 함께 소비
            else:
                out_t[name] = t[name]                    # dense/bias: 그대로(엔진이 F32 승격)
        save_file(out_t, str(out / shard.name))
        print(f"[{si}/{len(shards)}] {shard.name} ok (expert groups: {len(manifest['experts'])})")

    (out / "colibri_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done -> {out}  (expert groups quantized: {len(manifest['experts'])})")
    print("주의: gpt-oss는 glm_moe_dsa가 아님 → 엔진 어댑터 필요(docs/61).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="gpt-oss MXFP4 -> colibri int4 (prototype)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", help="gpt-oss HF 체크포인트 디렉토리")
    ap.add_argument("--out", help="출력 int4 스냅샷 디렉토리")
    ap.add_argument("--ebits", type=int, default=4, choices=(4, 8))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.model and args.out):
        ap.error("--selftest 또는 (--model 과 --out) 필요")
    return convert_checkpoint(args.model, args.out, args.ebits)


if __name__ == "__main__":
    raise SystemExit(main())
