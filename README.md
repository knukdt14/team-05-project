# 최종 RAG 퀴즈 생성 파이프라인

외부 실험에서 결정한 청킹 방법으로 교육자료 기반 퀴즈를 생성합니다.
이 디렉터리에는 청킹 방법을 비교·평가·선별하거나 자동 결정하는 기능이 없습니다.

## 사용 모델과 검색 설정

| 구분 | 설정 |
| --- | --- |
| 임베딩 | `BAAI/bge-m3` |
| 검색 | 슬라이드 중복 제거 검색 |
| Top-K | 5 |
| Fetch-K | 15 |
| 거리 | L2 |
| 생성 모델 | `Qwen/Qwen2.5-3B-Instruct` |

## 디렉터리

```text
integrated_final_pipeline/
├── pipeline_config.json
├── pyproject.toml
├── requirements.txt
├── results/
└── src/
    ├── documents.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    ├── quiz_generator.py
    ├── visualize_quizzes.py
    ├── settings.py
    └── main.py
```

## 설치

Python 3.11.x가 필요합니다. `requirements.txt`는 Python 자체를 설치하지
않으므로 Python 3.11로 가상환경을 생성해야 합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 퀴즈 생성

PPT 경로, 청킹 방법, 주제와 퀴즈 유형은 `src/main.py` 상단에
하드코딩되어 있습니다.

```python
PPTX_PATH = PROJECT_ROOT / "data" / "경북대 교육 발표자료 250416-1.pptx"
CHUNKING_STRATEGY = "sentence_pack"
QUIZ_TOPICS = [
    "중간날림의 원인",
]
QUIZ_TYPE = "multiple_choice"
VALIDATE_CHOICES = True
```

주제를 추가하려면 `QUIZ_TOPICS` 배열에 문자열을 추가합니다. 설정 후에는
명령행 옵션 없이 실행합니다.

```powershell
python src\main.py
```

생성 결과:

```text
results/generated_quizzes.json
```

객관식 `answer`는 보기 배열의 1부터 시작하는 번호입니다. 결과 객체는
`quiz_id`, `type`, `question`, `choices`, `answer`, `explanation`,
`evidence`, `source` 형식으로 저장됩니다.

## 객관식 복수정답 방지

객관식은 다음 검증을 거칩니다.

1. 정답 1개와 오답 3개를 생성하도록 요청합니다.
2. 보기 개수, 중복 여부와 정답 번호 범위를 코드로 검사합니다.
3. Qwen-3B가 검색 자료를 기준으로 각 보기의 정답 여부를 다시 확인합니다.
4. 정답 후보가 하나가 아니면 최대 3회 재생성합니다.

검증을 생략하려면 `main.py`의 `VALIDATE_CHOICES`를 `False`로 변경합니다.

## 생성 퀴즈 시각화

생성된 JSON을 문제 카드 형태의 HTML로 변환합니다.

```powershell
python src\visualize_quizzes.py
```

기본 입출력:

```text
입력: results/generated_quizzes.json
출력: results/generated_quizzes.html
```
