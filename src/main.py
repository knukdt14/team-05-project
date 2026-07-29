"""선택한 청킹 전략으로 BGE-M3 RAG 퀴즈를 생성한다."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from chunking import CHUNKING_CONFIGS, SUPPORTED_STRATEGIES, chunk_documents
from documents import extract_slide_structures, extract_slide_texts
from embeddings import BGEEmbedder
from quiz_generator import generate_quiz, load_llm
from retriever import retrieve_diverse
from settings import PROJECT_ROOT, load_config
from topic import extract_topics_llm, sample_topics
from vector_store import build_index


# 명령행 옵션을 생략했을 때 사용하는 기본 설정.
PPTX_PATH = PROJECT_ROOT / "data" / "경북대 교육 발표자료 250416-1.pptx"
CHUNKING_STRATEGY = "layout_aware"

# 주제는 main()에서 LLM(topic.py)으로 슬라이드를 요약해 추출한다 (Solar 재사용).
TOPIC_COUNT = 50
TOPIC_SEED = 10
QUIZ_TYPE = "multiple_choice"
VALIDATE_CHOICES = True


def prepare_pipeline(
    pptx_path: str | Path,
    strategy: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
):
    config = load_config()
    documents = (
        extract_slide_structures(pptx_path)
        if strategy in {"title_body", "layout_aware"}
        else extract_slide_texts(pptx_path)
    )
    chunks = chunk_documents(
        documents,
        strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        document_id=config["document_id"],
    )
    effective_size = (
        CHUNKING_CONFIGS[strategy]["chunk_size"]
        if chunk_size is None
        else chunk_size
    )
    effective_overlap = (
        CHUNKING_CONFIGS[strategy]["overlap"]
        if overlap is None
        else overlap
    )
    experiment_id = f"{strategy}_{effective_size}_{effective_overlap}"
    print(f"임베딩 모델 로딩: {config['embedding_model']}")
    embedder = BGEEmbedder(config["embedding_model"])
    collection = build_index(
        chunks,
        embedder,
        collection_name=f"final_{experiment_id}",
        persist_dir=PROJECT_ROOT / "results" / "chroma" / experiment_id,
        space=config["distance"],
    )
    print(f"인덱스 완료: {len(documents)} slides -> {len(chunks)} chunks")
    # 인덱싱이 끝난 뒤 BGE-M3를 CPU로 옮겨 Qwen-3B가 GPU 메모리를
    # 확보하게 한다. 이후 주제 검색용 query 임베딩만 CPU에서 계산한다.
    embedder.to("cpu")
    return config, embedder, collection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="청킹 전략을 선택해 RAG 퀴즈를 생성합니다."
    )
    parser.add_argument("--pptx", type=Path, default=PPTX_PATH)
    parser.add_argument(
        "--chunking-strategy",
        choices=SUPPORTED_STRATEGIES,
        default=CHUNKING_STRATEGY,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="지정하지 않으면 전략별 기본값을 사용합니다.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="recursive는 문자 수, 나머지 전략은 의미 단위 개수입니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="기본값: results/generated_quizzes_<전략>_<크기>_<겹침>.json",
    )
    parser.add_argument("--topic-count", type=int, default=TOPIC_COUNT)
    parser.add_argument("--topic-seed", type=int, default=TOPIC_SEED)
    return parser.parse_args(argv)


def _experiment_paths(
    strategy: str,
    chunk_size: int,
    overlap: int,
    output: Path | None,
) -> tuple[Path, Path]:
    experiment_id = f"{strategy}_{chunk_size}_{overlap}"
    output_path = (
        output
        if output is not None
        else PROJECT_ROOT / "results" / f"generated_quizzes_{experiment_id}.json"
    )
    failure_path = (
        output_path.parent / f"{output_path.stem}_failures.json"
    )
    return output_path, failure_path


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    strategy = args.chunking_strategy
    default_config = CHUNKING_CONFIGS[strategy]
    chunk_size = (
        default_config["chunk_size"] if args.chunk_size is None else args.chunk_size
    )
    overlap = default_config["overlap"] if args.overlap is None else args.overlap
    output_path, failure_log = _experiment_paths(
        strategy,
        chunk_size,
        overlap,
        args.output,
    )

    if not args.pptx.exists():
        raise FileNotFoundError(f"PPTX 파일이 없습니다: {args.pptx}")
    if args.topic_count <= 0:
        raise ValueError("topic-count는 1 이상이어야 합니다.")

    print(f"PPTX: {args.pptx}")
    print(f"청킹 방법: {strategy} (size={chunk_size}, overlap={overlap})")
    print(f"결과 파일: {output_path}")
    config, embedder, collection = prepare_pipeline(
        args.pptx,
        strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print(f"생성 모델 로딩: {config['llm_model']}")
    generator = load_llm(config["llm_model"])

    # LLM으로 슬라이드에서 주제 추출(캐시 있으면 재사용) 후 샘플링
    topic_pool = extract_topics_llm(args.pptx, generator, use_cache=True)
    quiz_topics = sample_topics(
        topic_pool,
        n=args.topic_count,
        seed=args.topic_seed,
    )
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
        print("저장:", save_results(quizzes, output_path))

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
            print("실패 목록:", save_failures(failures, failure_log))
            continue

    print(f"\n{'=' * 60}")
    print(f"완료: 생성 {len(quizzes)}개 / 실패 {len(failures)}개 (전체 {len(quiz_topics)}개)")
    if failures:
        print(f"실패 주제: {[item['topic'] for item in failures]}")
        print(f"실패 상세 로그: {failure_log}")
    print("=" * 60)


if __name__ == "__main__":
    main()
