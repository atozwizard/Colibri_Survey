# OLMoE 참고자료 출처

colibri의 **스트리밍 코어 검증 발판**(`external/colibri/c/olmoe.c`, `tools/convert_olmoe.py`)이 대상으로 삼는 실제 모델.

## 파일
- `paper-olmoe-arxiv-2409.02060.txt` — OLMoE: Open Mixture-of-Experts Language Models (arXiv:2409.02060), 전문 텍스트.

## 핵심 사실
- Allen AI(AI2), 2024-09 공개. **6.9B 총 파라미터 / 1.3B 활성**.
- **64 expert/layer, top-8 라우팅**(fine-grained), dropless token-choice routing.
- 5.1T 토큰 사전학습. 완전 오픈(가중치·데이터·코드·로그).
- 유사 활성규모(~1B) 대비 SOTA, Llama2-13B-Chat·DeepSeekMoE-16B 상회.

## 링크
- Paper: https://arxiv.org/abs/2409.02060
- Model: https://huggingface.co/allenai/OLMoE-1B-7B-0924
- Code: https://github.com/allenai/OLMoE
- transformers 문서: https://huggingface.co/docs/transformers/main/model_doc/olmoe

## colibri에서의 위상
- **골자가 아님**. GLM-5.2(`glm.c`)가 주 타깃이고 OLMoE는 스트리밍 코어를 작은 모델로 먼저 검증하는 Stage A 발판.
- 상세: `docs/80-olmoe-and-h100-recommendations.md` Q1.
