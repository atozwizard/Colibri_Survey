# GLM-5.2 정리 노트

## 기본 사양
- 개발: Zhipu AI (Z.ai / zai-org). 계열: GLM-5 → GLM-5.1 → GLM-5.2.
- 구조: Sparse Mixture-of-Experts. **744B(≈753B) 총 파라미터, 토큰당 약 40B 활성** (744B-A40B).
- 컨텍스트: **1M 토큰**. 라이선스: **MIT**(가중치). 배포 정밀도: BF16, FP8.
- 사전학습 28.5T 토큰. 코딩·장기(long-horizon) 에이전트 작업 지향.
- arXiv 기술 리포트: 2602.15763.

## 핵심 아키텍처 요소 (colibri와 직접 연결)
1. **DSA (DeepSeek Sparse Attention)**: 모든 토큰이 아니라 학습된 희소 부분집합에만 attend → 장문 비용 절감. GLM-5는 DSA를 통합해 배포 비용 감소.
2. **IndexShare**: 4개 sparse attention 레이어마다 lightweight indexer 1개를 공유(첫 레이어에 배치, top-k 인덱스를 4레이어 재사용). 1M 컨텍스트에서 per-token FLOPs를 2.9× 감소.
3. **MTP (Multi-Token Prediction) 레이어**: speculative decoding용. draft model로서 비용 최소화 + acceptance 최대화가 목표. GLM-5.2에서 accepted length를 최대 20% 향상. rejection sampling 도입, end-to-end TV loss로 학습.

## colibri와의 관계
- colibri는 GLM-5.2의 위 요소(DSA, MLA류 압축 KV, MTP)를 순수 C로 충실히 재현하고, 여기에 "expert 디스크 스트리밍"을 얹어 소비자 하드웨어 구동을 가능케 함.
- colibri README에서 언급하는 `glm_moe_dsa` 참조 아키텍처가 이 GLM-5.2 모델에 해당.

## 파생 (colibri용 int4 변환본)
- FP8 원본: huggingface.co/zai-org/GLM-5.2-FP8
- colibri용 int4(+int8 MTP head): huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp
