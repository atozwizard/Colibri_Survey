# gpt-oss-20b weight 레이아웃 (검증됨)

`openai/gpt-oss-20b`의 safetensors 헤더를 HTTP range-fetch로 직접 확인(2026-07). 변환기(`scripts/mxfp4_to_int4_prototype.py`)의 근거.

## config 요약 (`gpt-oss-20b-config.json`)
- `architectures`: `GptOssForCausalLM`, `model_type`: `gpt_oss`
- layers 24, hidden 2880, intermediate 2880, head_dim 64, heads 64, kv_heads 8 (GQA)
- **num_local_experts 32, num_experts_per_tok 4**
- attention: `sliding_attention`/`full_attention` 교대, `sliding_window` 128, rope `yarn`(factor 32), theta 150000
- `swiglu_limit` 7.0
- **quantization**: `quant_method: mxfp4`, `modules_to_not_convert`:
  `self_attn`, `mlp.router`, `embed_tokens`, `lm_head` → **MXFP4는 expert MLP만**.

## expert 텐서 (per layer, 32 experts grouped)
safetensors 헤더 실측:

| key | dtype | shape | 의미 |
|---|---|---|---|
| `...experts.gate_up_proj_blocks` | U8 | `[32, 5760, 90, 16]` | E×O×(K/32)×16B; K=2880(hidden), O=5760=2×inter |
| `...experts.gate_up_proj_scales` | U8 | `[32, 5760, 90]` | (E,O,block)당 1 E8M0 |
| `...experts.gate_up_proj_bias` | BF16 | `[32, 5760]` | |
| `...experts.down_proj_blocks` | U8 | `[32, 2880, 90, 16]` | K=2880(inter), O=2880(hidden) |
| `...experts.down_proj_scales` | U8 | `[32, 2880, 90]` | |
| `...experts.down_proj_bias` | BF16 | `[32, 2880]` | |

### 블록 산술
- 16 byte/block × 2 nibble/byte = **32 element/block**.
- 90 block × 32 = **2880 = K**(입력차원). ✓
- 스케일은 (expert, out_row, block)마다 1개 = **행 내 K를 32씩 그룹 스케일**.

## 변환 규칙 (→ colibri int4)
1. expert별로 `[O, nb=90, 16]` blocks + `[O, 90]` scales → f32 `[O, K]`
   - nibble→E2M1 크기 LUT `[0,.5,1,1.5,2,3,4,6]`(부호 별도), × `2^(E8M0-127)`.
2. f32 `[O, K]` → colibri `quant_int4`(행 amax→scale, nibble[-8,7], pack) → `name`(U8)+`name.qs`(F32).
3. bias(bf16)·router·attn·embed·lm_head는 비-MXFP4 → 그대로.

## 남은 확인
- `gate_up`의 gate/up 인터리브 순서(엔진 어댑터에서 SwiGLU 적용 시 분리 규칙).
- nibble의 원소 순서(byte 내 lo=2j, hi=2j+1)는 표준 가정; 실 디코드로 최종 검증 권장.

## 출처
- `https://huggingface.co/openai/gpt-oss-20b` (config.json, model.safetensors.index.json, safetensors 헤더 range-fetch)
- 변환기: `scripts/mxfp4_to_int4_prototype.py`
