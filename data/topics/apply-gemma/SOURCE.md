# data/topics/apply-gemma — 출처

| # | 자료 | URL | 취득일 | 저장 형태 |
|---|---|---|---|---|
| 1 | Gemma 4 기술 리포트 | https://arxiv.org/abs/2607.02770 | 2026-07-13 | 원문 텍스트 `gemma4-tech-report-arxiv-2607.02770.txt` |
| 2 | Gemma 4 모델 카드 (Google AI) | https://ai.google.dev/gemma/docs/core/model_card_4 | 2026-07-13 | 링크 |
| 3 | Gemma 4 소개 블로그 (Google) | https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ | 2026-07-13 | 링크 |
| 4 | Gemma4 (Hugging Face transformers) | https://huggingface.co/docs/transformers/model_doc/gemma4 | 2026-07-13 | 링크 |

## 확인된 핵심 사양
- **31B = Dense**(expert 없음) → colibrì 스트리밍 부적합.
- **26B-A4B = MoE**: 25.2B 총 / 3.8B 활성, 30층, 128 expert(8 활성 + 1 shared), sliding window 1024, 컨텍스트 256K, 멀티모달(text+image, 비전 인코더 ~550M). Q4_0 4bit ≈14.4GB. Apache-2.0.
- 그 외 사이즈: 12B/E4B/E2B(모두 Dense).
- 설계 문서: `../../../docs/62-apply-gemma4.md`.
