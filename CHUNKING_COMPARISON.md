# PPT 청킹 비교

임베딩 모델, Chroma, 검색 질의와 `top_k`는 고정하고 청킹 방식만
비교한다.

| 전략 | 설정 |
|---|---|
| `recursive` | 150자 + 이전 청크 끝 30자 overlap |
| `sentence_pack` | 300자 + 이전 문장/줄 1개 overlap |
| `slide_aware` | 기존 평탄 텍스트 추출을 유지하고 첫 줄을 제목으로 반복하며, 슬라이드 안에서만 줄·문장 경계로 분할 |
| `title_body` | `documents.py`에서 제목·본문 도형을 구조화하고, 본문이 300자를 넘을 때만 줄·문장 경계로 분할 |

## 청크 통계

```powershell
python src/compare_chunking.py `
  "경북대 교육 발표자료 250416-1.pptx" `
  --stats-only
```

## Chroma 검색 비교

```powershell
pip install -r requirements.txt

python src/compare_chunking.py `
  "경북대 교육 발표자료 250416-1.pptx" `
  --top-k 3 `
  --embedding-model "intfloat/multilingual-e5-small" `
  --output "chunking_results.json"
```

실행 결과에는 전략별 청크 수, 평균 길이, `recall@k`, MRR,
`context_precision@k`와 질의별 검색 슬라이드가 기록된다.

자체 평가 질의는 JSON 또는 JSONL 파일로 만들어 `--eval-file`에 전달한다.

```json
[
  {
    "query": "위험성 평가의 주요 절차는 무엇인가?",
    "relevant_slides": [25, 26]
  }
]
```
