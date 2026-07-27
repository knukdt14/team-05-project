# 베이스라인 RAG 파이프라인 (아진산업 신입사원 안전교육 퀴즈)

## 요청하신 설정값 반영 현황

| 항목 | 설정값 | 반영 위치 |
|---|---|---|
| PPT 텍스트 추출 | chunk_size=150 | `src/chunking.py` |
| 임베딩 모델 | HF 중 가장 빠르고 적합한 모델 | 아래 "임베딩 모델 선택 근거" 참고 |
| 벡터 DB | Chroma | `src/index.py` |
| LLM | Qwen | `src/generator.py` (Qwen2.5-1.5B-Instruct) |
| 평가지표 | grounded_score, recall@k, BERTScore | `src/evaluator.py`, `src/retriever.py` |

## 임베딩 모델 선택 근거: `intfloat/multilingual-e5-small`

기존에 얘기했던 `paraphrase-multilingual-MiniLM-L12-v2` 대신 이걸 추천함:

- 파라미터 규모(118M)는 비슷해서 **속도는 동급**
- e5 계열은 애초에 **검색(retrieval) 목적으로 대조학습**된 모델이라, RAG의 retriever
  용도에는 범용 문장유사도 모델(MiniLM-paraphrase)보다 더 적합
- MTEB 다국어 리트리벌 벤치마크에서 같은 체급 대비 점수가 더 높은 편
- 주의점: 검색 성능을 제대로 내려면 입력 앞에 `"query: "` / `"passage: "` 접두사를
  붙여야 함 (`embeddings.py`의 `ProductionEmbedder`에 이미 반영)

## 평가지표 추천 (grounded_score, recall@k, BERTScore 외 추가)

요청하신 3개에 아래 2개를 추가하는 걸 추천함 — 이유:

1. **MRR (Mean Reciprocal Rank)** — recall@k는 "정답이 top-k 안에 들어왔는지"만
   보는데, MRR은 "몇 번째 순위에 나왔는지"까지 반영함. 청킹/top-k 실험에서
   recall은 똑같이 1.0이 나와도 MRR로 순위 품질 차이를 구분할 수 있어서, recall@k와
   짝을 이루는 지표로 추천.
2. **Context Precision** — recall@k의 반대 짝. "가져온 top-k 중 실제로 관련있는
   비율"을 봄. 청크를 크게 자르면 recall은 오르지만 precision은 떨어지는 트레이드
   오프가 실제로 나타나는데, 이 두 지표를 같이 봐야 그 트레이드오프를 보여줄 수 있음.

(참고로 RAGAS의 Faithfulness/Answer Relevancy도 있는데, 이건 LLM을 판정자로
따로 호출해야 해서 - 별도 API 비용/설정이 필요함. 시간 되면 추가하고, 아니면
grounded_score가 유사한 역할(근거성 체크)을 이미 하고 있어서 없어도 무방함.)

## 파일 구성

```
rag_baseline/
├── requirements.txt
├── src/
│   ├── documents.py    # pptx -> 슬라이드별 텍스트 추출
│   ├── chunking.py      # chunk_size=150 recursive 청킹
│   ├── embeddings.py    # ProductionEmbedder(실사용) / LocalTestEmbedder(로컬검증)
│   ├── index.py         # Chroma 인덱싱
│   ├── retriever.py     # top-k 검색 + recall@k, MRR, context_precision
│   ├── generator.py     # Qwen2.5-1.5B-Instruct baseline/RAG 생성
│   ├── evaluator.py     # grounded_score, bert_score_eval
│   └── run_demo.py      # 전체 파이프라인 실행 진입점
```

## 실행 방법

### 지금 이 샌드박스에서 실행 가능한 부분 (실제로 테스트 완료됨)
```bash
cd src
python3 run_demo.py
```
huggingface.co 접속이 막힌 환경이라 `LocalTestEmbedder`(TF-IDF 기반, 모델 다운로드
불필요)로 대체해서 **청킹 -> 인덱싱 -> 검색 -> 평가**까지 배관이 정상 작동하는 걸
확인함. 실제 실행 결과:

```
1) 문서 추출 + 청킹 (chunk_size=150)
슬라이드 165개 -> 청크 465개

3) 검색 평가 (recall@3, MRR, context_precision@3)
- 크레인 작업 시 위험발생요인과 대책은?  | recall@3=1.00 MRR=1.00 precision@3=0.33
- 프레스 작업 중 재해사례는 언제 어디서 발생했나? | recall@3=1.00 MRR=1.00 precision@3=0.33
...
평균 recall@3=1.000  평균 MRR=1.000  평균 precision@3=0.333

4) grounded_score 예시
실제근거 기반 답변 grounded_score: 0.329
지어낸 답변 grounded_score      : 0.099
```
(recall/MRR이 다 1.0인 건 TF-IDF조차 안전관리 섹션 키워드가 명확해서 잘 맞춘 것.
실제 모델로 바꾸면 더 어려운 질문에서 차이가 드러날 것으로 예상)

### 인터넷 되는 환경(다른 AI/로컬 PC/GPU서버)에서 마저 실행할 부분
```python
# embeddings.py에서 한 줄만 교체
from embeddings import ProductionEmbedder
embedder = ProductionEmbedder("intfloat/multilingual-e5-small")
```
그리고 `generator.py`(Qwen2.5-1.5B-Instruct 로딩 + 생성), `evaluator.bert_score_eval`
(BERTScore)을 실행하면 전체 파이프라인 완성.

## 한계 및 다음 단계
- 지금 recall/MRR 결과는 TF-IDF 임시 임베딩 기준이라 실제 의미기반 검색
  성능이 아님 — 실제 모델(e5-small) 연결 후 재측정 필요
- 질문-정답 세트(`EVAL_SET`)가 5개뿐이라 통계적으로 부족함 — 안전관리 섹션 전체
  (슬라이드 34~42)에서 15~20개 이상으로 확장 권장
- BERTScore/grounded_score는 "생성된 답변"이 있어야 계산되는데, 지금은 Qwen 생성이
  안 된 상태라 grounded_score도 예시 텍스트로만 검증함 — 실제 생성 결과 넣어서
  재실행 필요
