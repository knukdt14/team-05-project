# B(임베딩/검색) 최종 결과 보고서

담당: 임베딩 모델 선정 및 검색(retrieval) 최적화
대상 자료: 경북대_교육_발표자료_250416-1.pptx (아진산업 신입사원 교육자료, 165슬라이드)

---

## 1. 평가셋 구성

- C의 실제 퀴즈 생성 스키마(`quiz_id, type, question, choices, answer, explanation, evidence, source`)와 동일한 포맷으로 **49개 문항** 직접 제작
- 안전관리/생산관리/품질보증/부품개발/자동화기술/전동화개발팀 등 **14개 섹션**에 걸쳐 골고루 구성 (특정 섹션에 치우치지 않도록)
- 문항당 정답 슬라이드 1개로 설계 (파일: `data/eval/ground_truth_quiz.py`, `.jsonl`)

## 2. 임베딩 모델 5개 비교

| 후보 | 비고 |
|---|---|
| dragonkue/multilingual-e5-small-ko-v2 | 한국어 파인튜닝된 e5, 118M |
| intfloat/multilingual-e5-small | 원본 다국어 e5, 118M |
| **BAAI/bge-m3** | 다국어+검색특화, 560M |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 문장유사도(STS)용, 118M — 비교군 |
| jhgan/ko-sroberta-multitask | 한국어 STS용, 비교군 |

선정 전 공개 벤치마크(MTEB 한국어 검색 태스크)로 1차 스크리닝 → paraphrase-MiniLM은 문서검색 태스크에서 이미 낮은 점수(0.41)로 확인되었으나, 검증 목적상 5개 모두 실측 비교 진행.

### 결과 (k=5, retrieve_diverse 적용 후)

| 모델 | recall@5 | MRR | precision@5 | diversity@5 | 초/질문 |
|---|---|---|---|---|---|
| dragonkue-e5-small-ko-v2 | 0.939 | 0.829 | 0.188 | 1.000 | 0.011 |
| intfloat/e5-small | 0.939 | 0.824 | 0.188 | 1.000 | 0.011 |
| **BAAI/bge-m3** | **0.980** | **0.896** | **0.196** | 1.000 | 0.017 |
| paraphrase-MiniLM | 0.714 | 0.536 | 0.143 | 1.000 | 0.010 |
| ko-sroberta | 0.878 | 0.737 | 0.176 | 1.000 | 0.010 |

**종합 판단(단일 지표 편향 방지)**: MRR 단독, recall+MRR 단순평균, recall 가중(0.7) 세 가지 방식으로 각각 재계산해도 **BAAI/bge-m3가 전부 1위** → 특정 지표 선택에 좌우되지 않는 안정적 결론.

| 방식 | bge-m3 점수 | 2위 모델 점수 |
|---|---|---|
| MRR 단독 | 0.896 | 0.829 (dragonkue) |
| recall+MRR 평균 | 0.938 | 0.884 (dragonkue) |
| recall 가중(0.7) | 0.955 | 0.906 (dragonkue) |

## 3. 이슈 발견 및 해결: top-k 안 슬라이드 중복

**발견**: 5개 모델 전부 precision@k가 이론적 최댓값(1/k) 근처에 몰려 변별력이 없어 보였음. 원인 조사 결과, top-k 검색 시 **같은 슬라이드에서 나온 여러 청크가 자리를 중복 차지**하는 문제 확인 (예: `[38, 38, 39]`처럼 상위 3개 중 2개가 같은 슬라이드).

**정량화**: `diversity@k`(top-k 내 서로 다른 슬라이드 비율) 지표를 새로 정의해 측정한 결과, 수정 전 5개 모델 전부 **0.73~0.83 수준**으로 낮게 나옴 (모델과 무관한 공통 이슈로 확인).

**해결**: `retriever.py`에 `retrieve_diverse()` 추가 — 후보군(`fetch_k`)을 넉넉히 가져온 뒤, 같은 슬라이드는 1개만 채택하고 다음 순위로 넘어가는 방식으로 재구성.

**검증**: k=5 고정 기준, 수정 전/후 비교 (로직만 변경, 다른 변수 통제):

| 모델 | diversity (전→후) | recall (전→후) | MRR (전→후) |
|---|---|---|---|
| dragonkue-ko-v2 | 0.743 → 1.000 | 0.918 → 0.939 ↑ | 0.825 → 0.829 ↑ |
| e5-small | 0.739 → 1.000 | 0.939 → 0.939 - | 0.822 → 0.824 ↑ |
| bge-m3 | 0.731 → 1.000 | 0.980 → 0.980 - | 0.893 → 0.896 ↑ |
| paraphrase-MiniLM | 0.833 → 1.000 | 0.714 → 0.714 - | 0.527 → 0.536 ↑ |
| ko-sroberta | 0.751 → 1.000 | 0.837 → 0.878 ↑ | 0.723 → 0.737 ↑ |

**5개 모델 전부 예외 없이 diversity 개선 + recall/MRR 손해 없음** → 특정 모델에 우연히 나타난 효과가 아니라 구조적 개선임을 확인.

**한계 발견**: 이 해결책은 `fetch_k`가 `k`보다 충분히 커야만 유효함. top-k 스윕 실험(k=3/5/10/15, fetch_k=15 고정)에서 k=10부터 diversity가 다시 하락(0.920 → 0.676)하는 것을 확인 — 후보군 자체가 부족해지면 중복 제거 로직이 목표 개수를 못 채우기 때문. **fetch_k는 항상 k의 약 3배 이상을 유지해야 함**.

## 4. BAAI/bge-m3 파라미터 최적화

선정된 모델 하나로 3가지 실험 진행 (`optimize_bge_m3.py`):

### 실험 A: top-k 스윕 (fetch_k=15 고정)
| k | recall | MRR | diversity |
|---|---|---|---|
| 3 | 0.959 | 0.891 | 1.000 |
| 5 | 0.980 | 0.896 | 1.000 |
| 10 | 0.980 | 0.896 | 0.920 |
| 15 | 0.980 | 0.896 | 0.676 |

→ k=5부터 recall/MRR 수확체감, k=10 이상은 fetch_k 부족으로 diversity 저하만 유발. **k=5가 최적**.

### 실험 B: fetch_k 스윕 (k=5 고정)
| fetch_k | precision | diversity |
|---|---|---|
| 7 | 0.220 | 0.910 |
| 10 | 0.199 | 0.988 |
| 15 | 0.196 | 1.000 |
| 20, 30 | 0.196 | 1.000 |

→ fetch_k=15부터 diversity 1.000 완전 달성, 그 이상은 효과 동일. **fetch_k=15가 최적(효율적인 지점)**.

### 실험 C: 거리계산 방식 비교 (k=5, fetch_k=15)
| space | recall | MRR |
|---|---|---|
| l2 | 0.980 | 0.896 |
| cosine | 0.980 | 0.896 |

→ 완전 동일한 결과 (임베딩이 정규화되어 있어 이론과 일치). **기본값(l2) 유지로 결정**.

## 5. 최종 확정 파라미터

| 파라미터 | 값 | 근거 |
|---|---|---|
| 임베딩 모델 | BAAI/bge-m3 | 5개 모델 중 모든 종합 판단 방식에서 1위 |
| 검색 로직 | retrieve_diverse (중복 제거) | diversity 0.73~0.83 → 1.000, 부작용 없음 |
| top-k | 5 | recall/MRR 수확체감 지점, 그 이상은 이득 없이 diversity만 저하 |
| fetch_k | 15 (k의 3배) | 그 미만은 diversity 미달, 그 이상은 효과 동일 |
| 거리계산 방식 | l2 (기본값) | cosine과 결과 동일, 변경 불필요 |

## 6. A(청킹) 담당자에게: 통합 규약

본 B 파트의 모든 스크립트는 `chunking.py`가 아래 인터페이스를 유지한다는 전제로 작성됨:

```python
def chunk_documents(docs: list[dict]) -> list[dict]:
    # 입력: [{"slide_no": int, "text": str}, ...]
    # 출력: [{"chunk_id": str, "slide_no": int, "text": str}, ...]
```

**청킹 전략(chunk_size, overlap, 분할방식 등)이 바뀌어도 위 함수명과 입출력 형태만 유지되면 B의 모든 코드(index.py, retriever.py, compare_models_*.py, optimize_bge_m3.py)는 수정 없이 그대로 재사용 가능함.** 청킹이 바뀌면 `data/eval/ground_truth_quiz.py`의 `chunk_id` 필드만 재검증 필요 (현재는 A의 정식 청킹 결과가 없어 임시 추정값으로 채워둔 상태).

## 7. 업로드 파일 목록

```
src/
├── documents.py          (공용 - 그대로 유지, A/B 공통 사용)
├── embeddings.py          ★ B 신규/수정
├── index.py               ★ B 수정 (space 파라미터 추가)
├── retriever.py           ★ B 핵심 수정 (retrieve_diverse 추가)
├── compare_models_before.py  ★ B 신규
├── compare_models_after.py   ★ B 신규
├── optimize_bge_m3.py         ★ B 신규
data/eval/
├── ground_truth_quiz.py   ★ B 신규 (49문항 정답셋)
├── ground_truth_quiz.jsonl
requirements.txt           ★ B 관련 의존성 업데이트 필요
```
