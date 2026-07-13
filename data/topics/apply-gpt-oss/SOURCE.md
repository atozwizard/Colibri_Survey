# data/topics/apply-gpt-oss — 출처

| # | 자료 | URL | 취득일 | 저장 형태 |
|---|---|---|---|---|
| 1 | gpt-oss-120b & gpt-oss-20b 모델카드 | https://arxiv.org/abs/2508.10925 | 2026-07-13 | 원문 텍스트 `model-card-arxiv-2508.10925.txt` |
| 2 | Introducing gpt-oss (OpenAI) | https://openai.com/index/introducing-gpt-oss/ | 2026-07-13 | 링크 |
| 3 | GptOss (Hugging Face transformers 문서) | https://huggingface.co/docs/transformers/model_doc/gpt_oss | 2026-07-13 | 링크 |
| 4 | GPT OSS (NVIDIA Megatron Bridge) | https://docs.nvidia.com/nemo/megatron-bridge/latest/models/gpt_oss/gpt-oss.html | 2026-07-13 | 링크 |

## 확인된 핵심 사양(20b)
- MoE 20.9B 총 / 3.6B 활성, 24층, hidden 2880, 32 expert top-4, GQA(64Q/8KV, head_dim 64),
  attention sinks(softmax bias), sliding window(128)↔full 교대, RoPE+YaRN(128k), RMSNorm, QuickGELU-gated, MXFP4(4.25bit), vocab 201,088. Apache-2.0.
- 설계 문서: `../../../docs/61-apply-gpt-oss-20b.md`.
