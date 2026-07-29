"""Final RAG quiz model: layout-aware chunking + Upstage embeddings."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from chunking import CHUNKING_CONFIGS, chunk_documents
from documents import extract_slide_structures
from quiz_generator import generate_quiz, load_llm
from retriever import retrieve_diverse
from settings import PROJECT_ROOT, load_config
from topic import extract_topics_llm, sample_topics
from upstage_embeddings import UpstageEmbedder
from vector_store import build_index


CHUNKING_STRATEGY = "layout_aware"
CHUNK_SIZE = int(CHUNKING_CONFIGS[CHUNKING_STRATEGY]["chunk_size"])
OVERLAP = int(CHUNKING_CONFIGS[CHUNKING_STRATEGY]["overlap"])
TOPIC_COUNT = 50
TOPIC_SEED = 42
QUIZ_TYPE = "multiple_choice"
VALIDATE_CHOICES = True

PPTX_PATH = PROJECT_ROOT / "data" / "경북대 교육 발표자료 250416-1.pptx"
RESULTS_DIR = PROJECT_ROOT / "results_final"
DEFAULT_OUTPUT = RESULTS_DIR / "generated_quizzes_layout_aware_upstage.json"
FAILURE_LOG = RESULTS_DIR / "failed_topics_layout_aware_upstage.json"
EMBEDDING_CACHE = RESULTS_DIR / "cache" / "upstage_embeddings.json"
QUERY_EMBEDDING_CACHE = RESULTS_DIR / "cache" / "upstage_query_embeddings.json"
CHROMA_DIR = RESULTS_DIR / "chroma" / "layout_aware_upstage_400_1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx", type=Path, default=PPTX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topic-count", type=int, default=TOPIC_COUNT)
    parser.add_argument("--topic-seed", type=int, default=TOPIC_SEED)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.3,
        help="Upstage 문서 임베딩 API 요청 사이의 대기 시간(초)",
    )
    return parser.parse_args(argv)


def save_json(data, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def prepare_pipeline(
    pptx_path: Path,
    *,
    request_interval: float,
):
    config = load_config()

    # layout_aware가 제목 placeholder, 도형 순서, 표 헤더를 이용할 수 있도록
    # 평문 추출이 아닌 구조화 슬라이드 추출을 사용한다.
    documents = extract_slide_structures(pptx_path)
    chunks = chunk_documents(
        documents,
        CHUNKING_STRATEGY,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        document_id=config["document_id"],
    )

    print("임베딩: Upstage solar-embedding-1-large (passage/query 분리)")
    print(f"청크 {len(chunks)}개 임베딩 중")
    embedder = UpstageEmbedder(
        cache_path=EMBEDDING_CACHE,
        query_cache_path=QUERY_EMBEDDING_CACHE,
        request_interval=request_interval,
    )
    collection = build_index(
        chunks,
        embedder,
        collection_name="final_layout_aware_upstage_400_1",
        persist_dir=CHROMA_DIR,
        space=config["distance"],
    )
    print(f"인덱스 완료: {len(documents)} slides -> {len(chunks)} chunks")
    print(f"문서 임베딩 캐시: {EMBEDDING_CACHE}")
    print(f"질문 임베딩 캐시: {QUERY_EMBEDDING_CACHE}")
    return config, embedder, collection


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.pptx.exists():
        raise FileNotFoundError(f"PPTX 파일이 없습니다: {args.pptx}")
    if args.topic_count <= 0:
        raise ValueError("topic-count는 1 이상이어야 합니다.")
    if args.request_interval < 0:
        raise ValueError("request-interval은 0 이상이어야 합니다.")

    print("[최종 모델]")
    print(
        f"청킹={CHUNKING_STRATEGY} "
        f"(size={CHUNK_SIZE}, overlap={OVERLAP})"
    )
    print("임베딩=Upstage solar-embedding-1-large")
    print(f"PPTX={args.pptx}")
    print(f"출력={args.output}")

    config, embedder, collection = prepare_pipeline(
        args.pptx,
        request_interval=args.request_interval,
    )

    print(f"생성 모델: {config['llm_model']}")
    generator = load_llm(config["llm_model"])
    topic_pool = extract_topics_llm(args.pptx, generator, use_cache=True)
    quiz_topics = sample_topics(
        topic_pool,
        n=args.topic_count,
        seed=args.topic_seed,
    )
    if not quiz_topics:
        raise ValueError("추출된 주제가 없습니다.")
    print(f"주제 {len(quiz_topics)}개 (seed={args.topic_seed})")

    quizzes: list[dict] = []
    failures: list[dict] = []
    failure_path = (
        args.output.parent / f"{args.output.stem}_failures.json"
        if args.output != DEFAULT_OUTPUT
        else FAILURE_LOG
    )

    for index, topic in enumerate(quiz_topics, start=1):
        print(f"\n[{index}/{len(quiz_topics)}] 주제: {topic}")
        retrieved_output: list[dict] = []
        try:
            retrieved = retrieve_diverse(
                collection,
                embedder,
                topic,
                k=int(config["top_k"]),
                fetch_k=int(config["fetch_k"]),
            )
            retrieved_output = [
                {
                    "rank": rank,
                    "chunk_id": item["chunk_id"],
                    "slide_no": int(item["slide_no"]),
                    "title": str(item.get("title", "")),
                    "distance": float(item["distance"]),
                    "text": str(item["text"]),
                }
                for rank, item in enumerate(retrieved, start=1)
            ]
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
            quiz["topic"] = topic
            quiz["chunking"] = {
                "strategy": CHUNKING_STRATEGY,
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
            }
            quiz["embedding"] = {
                "provider": "upstage",
                "passage_model": "solar-embedding-1-large-passage",
                "query_model": "solar-embedding-1-large-query",
            }
            quiz["retrieved_chunks"] = retrieved_output
            quizzes.append(quiz)
            print("저장:", save_json(quizzes, args.output))
        except Exception as error:  # noqa: BLE001 - 실패를 기록하고 다음 주제로 진행
            print(f"  실패: {error!r}")
            failures.append(
                {
                    "topic": topic,
                    "error": str(error),
                    "retrieved_chunks": retrieved_output,
                    "traceback": traceback.format_exc(),
                }
            )
            save_json(failures, failure_path)

    # 이전 실행의 실패 로그가 성공한 재실행 뒤에도 남아 혼동을 주지 않도록
    # 정상 종료 시 현재 실행의 실패 목록(없으면 빈 배열)으로 갱신한다.
    save_json(failures, failure_path)
    print("\n" + "=" * 60)
    print(
        f"완료: 성공 {len(quizzes)}개 / 실패 {len(failures)}개 "
        f"(전체 {len(quiz_topics)}개)"
    )
    print(f"퀴즈 결과: {args.output}")
    if failures:
        print(f"실패 로그: {failure_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
