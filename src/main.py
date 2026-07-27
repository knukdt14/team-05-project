"""하드코딩된 설정으로 BGE-M3 RAG 퀴즈를 생성한다."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from chunking import chunk_documents
from documents import extract_slide_structures, extract_slide_texts
from embeddings import BGEEmbedder
from quiz_generator import generate_quiz, load_llm
from retriever import retrieve_diverse
from settings import PROJECT_ROOT, load_config
from topic import extract_topics_llm, sample_topics
from vector_store import build_index


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "generated_quizzes.json"
FAILURE_LOG = PROJECT_ROOT / "results" / "failed_topics.json"

# 실행 설정: 명령행 옵션 대신 이 값만 수정한다.
PPTX_PATH = PROJECT_ROOT / "data" / "경북대 교육 발표자료 250416-1.pptx"
CHUNKING_STRATEGY = "sentence_pack"

# 주제는 main()에서 LLM(topic.py)으로 슬라이드를 요약해 추출한다 (Solar 재사용).
TOPIC_COUNT = 15
TOPIC_SEED = 10
QUIZ_TYPE = "multiple_choice"
VALIDATE_CHOICES = True


def prepare_pipeline(pptx_path: str | Path, strategy: str):
    config = load_config()
    documents = (
        extract_slide_structures(pptx_path)
        if strategy == "title_body"
        else extract_slide_texts(pptx_path)
    )
    chunks = chunk_documents(
        documents,
        strategy,
        document_id=config["document_id"],
    )
    print(f"임베딩 모델 로딩: {config['embedding_model']}")
    embedder = BGEEmbedder(config["embedding_model"])
    collection = build_index(
        chunks,
        embedder,
        collection_name=f"final_{strategy}",
        persist_dir=PROJECT_ROOT / "results" / "chroma" / strategy,
        space=config["distance"],
    )
    print(f"인덱스 완료: {len(documents)} slides -> {len(chunks)} chunks")
    # 인덱싱이 끝난 뒤 BGE-M3를 CPU로 옮겨 Qwen-3B가 GPU 메모리를
    # 확보하게 한다. 이후 주제 검색용 query 임베딩만 CPU에서 계산한다.
    embedder.to("cpu")
    return config, embedder, collection


def save_results(quizzes: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(quizzes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def save_failures(failures: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    if not PPTX_PATH.exists():
        raise FileNotFoundError(f"PPTX 파일이 없습니다: {PPTX_PATH}")

    print(f"PPTX: {PPTX_PATH}")
    print(f"청킹 방법: {CHUNKING_STRATEGY}")
    config, embedder, collection = prepare_pipeline(
        PPTX_PATH,
        CHUNKING_STRATEGY,
    )

    print(f"생성 모델 로딩: {config['llm_model']}")
    generator = load_llm(config["llm_model"])

    # LLM으로 슬라이드에서 주제 추출(캐시 있으면 재사용) 후 샘플링
    topic_pool = extract_topics_llm(PPTX_PATH, generator, use_cache=True)
    quiz_topics = sample_topics(topic_pool, n=TOPIC_COUNT, seed=TOPIC_SEED)
    if not quiz_topics:
        raise ValueError("추출된 주제가 없습니다. topic.py를 확인하세요.")
    print(f"주제({len(quiz_topics)}개): {quiz_topics}")

    quizzes: list[dict] = []
    failures: list[dict] = []

    def create(topic: str) -> None:
        retrieved = retrieve_diverse(
            collection,
            embedder,
            topic,
            k=int(config["top_k"]),
            fetch_k=int(config["fetch_k"]),
        )
        quiz = generate_quiz(
            generator,
            topic,
            retrieved,
            quiz_id=f"quiz-{len(quizzes) + 1:03d}",
            quiz_type=QUIZ_TYPE,
            max_attempts=5,
            file_label=config["file_label"],
            validate_choices=VALIDATE_CHOICES,
        )
        quizzes.append(quiz)
        print(json.dumps(quiz, ensure_ascii=False, indent=2))
        print("저장:", save_results(quizzes, DEFAULT_OUTPUT))

    for index, topic in enumerate(quiz_topics, start=1):
        print(f"\n[{index}/{len(quiz_topics)}] 주제: {topic}")
        try:
            create(topic)
        except Exception as error:  # noqa: BLE001 - 실패한 주제는 기록하고 계속 진행
            print(f"  실패: {error!r}")
            failures.append({
                "topic": topic,
                "error": str(error),
                "traceback": traceback.format_exc(),
            })
            print("실패 목록:", save_failures(failures, FAILURE_LOG))
            continue

    print(f"\n{'=' * 60}")
    print(f"완료: 생성 {len(quizzes)}개 / 실패 {len(failures)}개 (전체 {len(quiz_topics)}개)")
    if failures:
        print(f"실패 주제: {[item['topic'] for item in failures]}")
        print(f"실패 상세 로그: {FAILURE_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
