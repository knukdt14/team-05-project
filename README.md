# 신입사원 업무교육 퀴즈 생성 RAG

사내 PowerPoint 교육자료를 근거로 신입사원용 객관식 퀴즈를 생성하는 RAG 프로젝트입니다.
최종 파이프라인은 PPT의 제목·도형·표 구조를 보존해 청킹하고, Upstage 임베딩으로 관련
슬라이드를 검색한 뒤 Solar LLM이 정답·해설·근거·출처를 포함한 퀴즈를 생성합니다.
별도로 BGE-M3 기반 청킹 비교 실험을 제공해 검색 품질도 검증합니다.

## 프로젝트 목표

- 교육자료에 없는 사실을 만들어 내지 않도록, 검색 문맥과 출처가 있는 퀴즈를 생성합니다.
- 신입사원이 실제 업무교육에서 활용할 수 있는 4지선다 퀴즈·정답·해설·근거를 JSON과 HTML로 제공합니다.
- PPT에 적합한 청킹 전략과 검색 품질을 비교하고, 최종 설정을 재현 가능한 실행 코드로 제공합니다.
- 자동 검증과 사람·AI 평가를 통해 퀴즈의 구조적 정확성, 명확성, 교육 적합성을 확인합니다.

## 프로젝트 팀 및 역할 분담

| 구분 | 이름 | 주요 역할 |
| --- | --- | --- |
| 팀장 | 김창조 | 파이프라인 통합, 데이터·청킹·벡터 검색 구성, 최종 실행 흐름 관리 |
| 팀원 | 박정민 | Solar LLM 연동, 퀴즈 생성 프롬프트·보기 검증·퀴즈 UI 구현 |
| 팀원 | 현가은 | 퀴즈 주제 추출, 평가 체계·결과 검증 및 문서화 |

모든 팀원은 생성 결과 교차 검수와 최종 실험 결과 검토에 함께 참여합니다.

## 최종 파이프라인 설정

`src/main_final.py`의 실제 실행 기본값입니다.

| 항목 | 최종 값 |
| --- | --- |
| Python | 3.11.x |
| 입력 자료 | `data/경북대 교육 발표자료 250416-1.pptx` |
| 청킹 | `layout_aware` (`chunk_size=400`, `overlap=1`) |
| 기본 생성 수 | 주제 50개 |
| 주제 샘플링 시드 | 42 |
| 문서·질문 임베딩 | Upstage `solar-embedding-1-large-passage` / `solar-embedding-1-large-query` |
| 생성·검증 모델 | Upstage `solar-pro` |
| 벡터 저장소 | ChromaDB |
| 거리 함수 | L2 |
| 검색 설정 | Fetch-K 15 → 슬라이드 중복 제거 → Top-K 5 |
| 퀴즈 형식 | 4지선다 객관식 |
| 보기 검증 | 사용 |

임베딩·생성 모델과 검색 설정은 `pipeline_config.json`, 최종 청킹·문항 수·시드는
`src/main_final.py`에서 관리합니다. `src/main.py`는 BGE-M3를 사용하는 청킹 비교용
기본 파이프라인입니다.

## 전체 흐름

### 1. 최종 퀴즈 생성: `main_final.py`

```text
PPTX
  → 슬라이드 텍스트·제목·표 구조 추출
  → layout_aware 청킹 (400/1)
  → Upstage passage 임베딩 및 로컬 캐시
  → Chroma 컬렉션 생성
  → 슬라이드별 주제 추출 또는 캐시 재사용
  → 시드 42로 최대 50개 주제 선택
  → Upstage query 임베딩 → Fetch-K 15 검색 → 슬라이드 중복 제거 → Top-K 5
  → Solar로 객관식 퀴즈 생성
  → 정답 유일성·보기 중복·정답 범위 검증 (실패 시 부정형 객관식 재시도)
  → 문항별 중간 JSON 저장 및 실패 로그 저장
  → 검토용 또는 풀이용 HTML 생성
```

`main_final.py`는 문서 임베딩과 질문 임베딩을 별도 캐시에 저장해 동일 입력의 API
재호출을 줄입니다. 한 주제에서 실패해도 다음 주제를 계속 처리하며, 성공 문항은 즉시
결과 JSON에 저장하고 실패 원인과 검색 청크는 별도 로그에 남깁니다.

### 2. 청킹 방식별 검색 성능 비교

```text
동일한 고정 질문 + 정답 슬라이드 번호
  → 청킹 방식별 PPTX 청킹·인덱싱
  → 모든 방식에 동일한 질문 검색
  → Recall@K·Precision@K·MRR 계산
  → 검색된 청크를 근거로 퀴즈 생성
  → 방식별 JSON + 전체 통합 JSON 저장
  → 비교 HTML 또는 Word 보고서 생성
```

검색 성능을 공정하게 비교하려면 청킹 방식마다 새 질문을 생성하지 않고
`data/evaluation_queries.json`의 같은 질문을 사용해야 합니다. 각 질문에는 평가용
`relevant_slides`가 들어 있으며, 모든 방식이 같은 질문·정답 슬라이드로 평가됩니다.

## 설치

프로젝트 루트에서 다음 명령을 실행합니다.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows CMD

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`의 PyTorch는 CUDA 12.8 빌드로 고정되어 있습니다. 해당 CUDA
환경이 없으면 사용 환경에 맞는 PyTorch를 먼저 설치한 뒤 나머지 의존성을
설치해야 합니다.

프로젝트 루트에 `.env` 파일을 만들고 Upstage API 키를 설정합니다.

```dotenv
UPSTAGE_API_KEY=발급받은_API_KEY
```

API 키는 저장소에 커밋하지 마세요.

## 빠른 실행

### 최종 모델: layout-aware + Upstage 임베딩

첨부된 Upstage 실험 코드를 기반으로 한 최종 실행 파일은 `src/main_final.py`입니다.
PPT 구조를 보존하는 `layout_aware` 청킹과 Upstage
`solar-embedding-1-large-passage/query`를 사용합니다.

```bat
python src\main_final.py
```

기본적으로 50문항을 생성하며 결과는 다음 위치에 저장됩니다.

```text
results_final/generated_quizzes_layout_aware_upstage.json
results_final/chroma/layout_aware_upstage_400_1/
results_final/cache/upstage_embeddings.json
results_final/cache/upstage_query_embeddings.json
```

임베딩 캐시는 같은 청크와 질문의 유료 API 재호출을 방지합니다. 첫 실행은 모든
청크를 API로 임베딩하므로 시간이 걸리며 사용량에 따른 비용이 발생할 수 있습니다.
대용량 문서 캐시와 소형 질문 캐시는 별도 파일로 저장됩니다. Windows가 캐시 파일을
일시적으로 잠근 경우에는 자동으로 재시도하며, 캐시 저장만 실패하더라도 현재
임베딩을 메모리에서 사용해 퀴즈 생성을 계속합니다.

### 기본값으로 퀴즈 50개 생성

```bat
python src\main.py
```

현재 기본값은 `layout_aware`, 크기 400, 겹침 1, 주제 50개입니다.

주요 결과는 다음 위치에 생성됩니다.

```text
results/generated_quizzes_layout_aware_400_1.json
results/generated_quizzes_layout_aware_400_1_failures.json  # 실패가 있을 때
results/chroma/layout_aware_400_1/
results/llm_topics.json
```

### 다른 청킹 방식으로 생성

```bat
python src\main.py --chunking-strategy slide_aware --topic-count 50
python src\main.py --chunking-strategy recursive --chunk-size 200 --overlap 40 --topic-count 50
python src\main.py --chunking-strategy title_body --chunk-size 350 --overlap 1 --topic-count 20
```

입력과 출력 경로도 직접 지정할 수 있습니다.

```bat
python src\main.py --pptx data\다른자료.pptx --output results\my_quizzes.json
```

전체 옵션은 다음 명령으로 확인합니다.

```bat
python src\main.py --help
```

## 청킹 방식

| 방식 | 기본 크기/겹침 | 처리 방식 | 적합한 자료 |
| --- | ---: | --- | --- |
| `recursive` | 150/30 | 문장 단위로 묶고 이전 청크 끝의 문자를 겹침 | 일반 텍스트, 짧은 문맥 |
| `sentence_pack` | 300/1 | 문장 단위로 크기에 맞춰 패킹하고 이전 문장 일부를 유지 | 문장 경계가 뚜렷한 자료 |
| `slide_aware` | 300/1 | 첫 줄을 슬라이드 제목으로 보고 모든 청크에 제목을 반복 | 제목 구조가 단순한 PPT |
| `title_body` | 300/1 | PPT 제목 placeholder와 본문 도형을 분리한 뒤 제목을 반복 | 제목 placeholder가 잘 구성된 PPT |
| `layout_aware` | 400/1 | 제목·도형 순서·표 구조를 보존하고 표 헤더를 각 행에 반복 | 표와 배치 정보가 중요한 PPT |

`overlap`의 의미는 방식마다 다릅니다. `recursive`에서는 문자 수이고, 나머지
방식에서는 이전 문장 또는 의미 단위의 개수입니다.

## 주제 추출과 재현성

`src/topic.py`는 각 슬라이드를 Solar로 요약해 짧은 퀴즈 주제를 만들고
`results/llm_topics.json`에 캐시합니다. 캐시가 있으면 다음 실행부터 API를 다시
호출하지 않습니다.

`--topic-seed`는 **캐시에 있는 주제 목록에서 어떤 주제를 선택할지** 고정합니다.
같은 주제와 같은 시드를 사용해도 LLM 호출 자체는 샘플링을 사용하므로 퀴즈 문장이
항상 완전히 같지는 않습니다. 주제 캐시를 새로 만들려면 기존 캐시를 별도로
백업하거나 제거한 뒤 다시 실행해야 합니다.

## 검색과 퀴즈 생성 세부 흐름

아래 흐름에서 최종 파이프라인(`main_final.py`)은 Upstage 임베딩을, 비교·실험
파이프라인(`main.py`)은 BGE-M3 임베딩을 사용합니다.

1. `documents.py`가 슬라이드 번호, 제목, 본문 도형, 표를 읽고 위에서 아래,
   왼쪽에서 오른쪽 순서로 정렬합니다.
2. `chunking.py`가 선택된 방식에 맞게 청크를 만들고
   `문서ID-p슬라이드-c청크` 형식의 `chunk_id`를 부여합니다.
3. `upstage_embeddings.py` 또는 `embeddings.py`가 문서와 질문을 임베딩합니다.
4. `vector_store.py`가 실험별 Chroma 컬렉션을 새로 구성합니다.
5. `retriever.py`가 먼저 15개를 가져온 뒤 같은 슬라이드의 중복 청크를 제거하여
   최대 5개 슬라이드를 반환합니다.
6. `quiz_generator.py`가 검색 문맥에서 핵심 사실을 고르고 객관식 문항을 만듭니다.
7. 코드 검증과 LLM 검증으로 보기 4개, 정답 번호 범위, 보기 중복, 단일 정답,
   정답 표현 노출 등을 확인합니다. 일반 문항 생성이 계속 실패하면 부정형
   객관식 생성도 시도합니다.
8. 결과에는 질문, 보기, 정답, 해설, 근거와 함께 생성에 사용한 슬라이드 및
   `chunk_id`가 `sources` 배열로 기록됩니다.

## 고정 질문으로 모든 청킹 방식 비교

기본 평가 질문 파일은 `data/evaluation_queries.json`입니다.

```json
[
  {
    "query_id": "q01",
    "query": "질문 내용",
    "relevant_slides": [15, 20]
  }
]
```

모든 청킹 방식을 기본 질문 전체에 적용합니다.

```bat
python src\generate_chunking_comparison.py
```

질문 수나 청킹 방식을 제한할 수도 있습니다.

```bat
python src\generate_chunking_comparison.py --query-count 10
python src\generate_chunking_comparison.py --strategies slide_aware layout_aware --query-count 10
```

결과 파일:

```text
results/chunking_comparison/recursive_150_30.json
results/chunking_comparison/sentence_pack_300_1.json
results/chunking_comparison/slide_aware_300_1.json
results/chunking_comparison/title_body_300_1.json
results/chunking_comparison/layout_aware_400_1.json
results/chunking_comparison/all_chunking_quizzes.json
```

통합 JSON의 각 퀴즈에는 질문별 `retrieved_chunks`가 포함됩니다. 각 청크에서 순위,
청크 ID, 슬라이드 번호, 제목, L2 거리, 원문을 확인할 수 있습니다.

### 검색 지표

| 지표 | 의미 |
| --- | --- |
| Recall@K | 정답 슬라이드 중 Top-K 결과에 포함된 비율 |
| Precision@K | Top-K 결과 중 정답 슬라이드인 비율 |
| MRR | 최초 정답 슬라이드 순위의 역수 |

현재 검색기는 슬라이드 중복을 제거하므로 한 질문의 결과에는 같은 슬라이드가
한 번만 포함됩니다. L2 거리는 작을수록 질문과 가까운 청크입니다.

## 결과 시각화

### 청킹 비교 HTML

```bat
python src\visualize_chunking_comparison.py
```

출력:

```text
results/chunking_comparison/all_chunking_quizzes.html
```

HTML은 질문 주제별로 결과를 묶고, 각 질문 안에서 청킹 방식별 퀴즈와 그 퀴즈
생성에 실제 사용된 청크를 함께 보여줍니다. 검색 지표 요약과 방식별 탭도
포함됩니다.

### 생성 결과 검토용 HTML

정답·해설·근거를 펼쳐 보는 카드형 문서입니다.

```bat
python src\visualize_quizzes.py --input results\generated_quizzes_layout_aware_400_1.json --output results\generated_quizzes.html
```

### 직접 풀어 보는 퀴즈 UI

보기를 선택하고 즉시 채점하는 단일 HTML 페이지입니다.

```bat
python src\quiz_ui.py --input results\generated_quizzes_layout_aware_400_1.json --output results\quiz_ui.html
```

생성된 HTML 파일은 별도 서버 없이 브라우저에서 바로 열 수 있습니다. 입력과 출력
파일을 명시하면 스크립트 기본 파일명에 영향을 받지 않아 가장 안전합니다.

## Word 검색 성능 보고서

통합 비교 JSON을 표와 해석이 포함된 Word 문서로 변환합니다.

```bat
python src\build_chunking_metrics_report.py
```

출력:

```text
results/chunking_comparison/청킹_방식별_검색성능_평가보고서.docx
```

다른 결과 파일을 사용할 수 있습니다.

```bat
python src\build_chunking_metrics_report.py --input results\chunking_comparison\all_chunking_quizzes.json --output results\chunking_report.docx
```

## 사람·AI 품질 평가

검색 지표와 별도로 퀴즈의 명확성과 교육 적합성을 평가하는 보조 흐름입니다.

```text
생성 퀴즈 JSON
  → 평가자 3명용 Excel 체크리스트
  → 각 평가자가 0/1 입력
  → Solar AI 평가 결과 생성
  → 사람 간 Fleiss' Kappa
  → AI와 사람 다수결 간 Cohen's Kappa
```

이 세 스크립트 중 `build_rating_checklist.py`와 `ai_judge.py`는 기본 입출력 경로를
`src` 폴더 기준 상대경로로 사용합니다. 기본값을 그대로 사용할 때는 다음처럼
실행합니다.

```bat
cd src
python build_rating_checklist.py ..\results\generated_quizzes_layout_aware_400_1.json
python ai_judge.py ..\results\generated_quizzes_layout_aware_400_1.json
cd ..
python src\kappa_eval.py results
```

평가자 세 명이 `results/checklist_rater1.xlsx`부터
`results/checklist_rater3.xlsx`까지의 점수 칸을 모두 채운 뒤 Kappa 계산을
실행해야 합니다. `src/evaluator.py`에는 생성 근거와 검색 청크의 문자열 유사도를
계산하는 `grounded_score`도 있으며, BERTScore 함수는 현재 실행용 구현이 아니라
환경 구성이 필요한 자리표시자입니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `src/main.py` | 단일 청킹 퀴즈 생성 파이프라인과 CLI |
| `src/main_final.py` | layout-aware + Upstage 임베딩 최종 파이프라인 |
| `src/upstage_embeddings.py` | Upstage passage/query 임베딩과 로컬 캐시 |
| `src/settings.py` | 프로젝트 루트와 JSON 설정 로딩 |
| `src/documents.py` | PPTX 슬라이드의 제목·본문·표 추출 |
| `src/chunking.py` | 5개 청킹 방식과 기본 파라미터 |
| `src/embeddings.py` | BGE-M3 문서·질문 임베딩 |
| `src/vector_store.py` | Chroma 컬렉션 생성과 저장 |
| `src/retriever.py` | 슬라이드 중복 제거 검색과 검색 지표 함수 |
| `src/topic.py` | Solar 기반 주제 추출, 캐시, 시드 샘플링 |
| `src/quiz_generator.py` | Solar 호출, 퀴즈 생성, 재시도, 보기 검증 |
| `src/generate_chunking_comparison.py` | 고정 질문 기반 청킹 비교 실험 |
| `src/visualize_chunking_comparison.py` | 통합 비교 JSON을 HTML로 변환 |
| `src/visualize_quizzes.py` | 단일 결과를 검토용 HTML로 변환 |
| `src/quiz_ui.py` | 단일 결과를 풀이·채점용 HTML로 변환 |
| `src/build_chunking_metrics_report.py` | 검색 성능 Word 보고서 생성 |
| `src/build_rating_checklist.py` | 사람 평가자용 Excel 3개 생성 |
| `src/ai_judge.py` | Solar 기반 퀴즈 품질 자동 평가 |
| `src/kappa_eval.py` | 사람 간, AI-사람 간 평가 일치도 계산 |
| `src/evaluator.py` | 문자열 기반 근거 일치 점수와 BERTScore 자리표시자 |
| `tests/test_chunking.py` | `layout_aware` 표 헤더 반복과 입력 검증 테스트 |
| `pipeline_config.json` | 모델, 검색, 문서 식별자 설정 |
| `data/evaluation_queries.json` | 모든 청킹 방식에 공통으로 적용하는 평가 질문 |

`verify_import.py`는 특정 PC의 절대경로가 들어 있는 과거 로컬 확인 스크립트이므로
현재 프로젝트 실행에는 사용하지 않습니다. `README_BASE.md` 역시 이전 설명을
보관한 파일이며 현재 사용법은 이 문서를 기준으로 합니다.

## 결과 JSON 구조

단일 생성 결과는 퀴즈 객체의 배열입니다.

```json
[
  {
    "quiz_id": "quiz-001",
    "type": "multiple_choice",
    "question": "질문",
    "choices": ["보기 1", "보기 2", "보기 3", "보기 4"],
    "answer": 2,
    "explanation": "해설",
    "evidence": "검색 문맥에 근거한 문장",
    "sources": [
      {
        "file": "자료명",
        "slide": 10,
        "chunk_id": "ajin_training_250416-p010-c01"
      }
    ]
  }
]
```

`answer`는 0부터 시작하는 배열 인덱스가 아니라 **1부터 시작하는 보기 번호**입니다.
청킹 비교 JSON은 위 필드에 `query_id`, `query`, `chunking`,
`retrieved_chunks`를 추가합니다.

## 테스트와 기본 점검

```bat
python tests\test_chunking.py
python src\main.py --help
python src\generate_chunking_comparison.py --help
```

테스트는 표준 라이브러리 `unittest` 기반이므로 별도의 테스트 패키지가 필요하지
않습니다. `pytest`를 사용하는 개발 환경에서는 `python -m pytest`로도 실행할 수
있습니다.

## 자주 발생하는 문제

- `UPSTAGE_API_KEY` 오류: 프로젝트 루트의 `.env`에 키가 있는지 확인합니다.
- PPTX를 찾지 못함: 기본 파일명이 다르면 `--pptx`로 정확한 경로를 전달합니다.
- GPU 메모리 부족: 인덱싱 후 임베딩 모델은 CPU로 이동하지만, BGE-M3 로딩 자체가
  부담될 수 있습니다. 다른 프로세스를 종료하거나 CPU용 PyTorch 환경을
  사용합니다.
- 질문 수가 예상보다 적음: 주제 캐시에 유효한 주제가 50개 미만이거나 일부 문항
  검증이 실패했을 수 있습니다. 결과 옆의 `_failures.json`을 확인합니다.
- 결과가 이전과 달라짐: `--topic-seed`는 주제 선택만 고정하고 Solar의 생성
  샘플링까지 고정하지 않습니다.
- 비교 지표가 없음: 평가 질문에 `relevant_slides`가 없으면 검색 결과와 퀴즈는
  저장되지만 Recall@K, Precision@K, MRR은 계산되지 않습니다.
- HTML이 열리지 않음: 먼저 입력 JSON이 생성되었는지 확인하고 `--input`과
  `--output`을 모두 명시해 다시 생성합니다.
