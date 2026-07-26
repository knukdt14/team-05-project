"""BAAI/bge-m3로 네 가지 청킹 방법을 평가하고 최종 방법을 선별한다.

임베딩·검색 조건은 팀원 B의 최종 실험값으로 고정한다.

- embedding model: BAAI/bge-m3
- normalized embeddings, query/passage prefix 미사용
- Chroma distance: l2
- retrieval: slide-diverse top_k=5, fetch_k=15

실행 결과는 터미널 표와 CSV·HTML로 저장한다. 이미지 파일은 생성하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any

from chunking import CHUNKING_CONFIGS, build_all_chunk_sets
from documents import extract_slide_structures, extract_slide_texts
from embeddings import ProductionEmbedder
from index import build_index
from retriever import context_precision, mrr, recall_at_k, retrieve_diverse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PPTX = PROJECT_ROOT / "경북대 교육 발표자료 250416-1.pptx"
DEFAULT_EVAL_FILE = PROJECT_ROOT / "eval_questions_100.json"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "result" / "chunking_selection_bge_m3.csv"
DEFAULT_HTML_OUTPUT = PROJECT_ROOT / "result" / "chunking_selection_bge_m3.html"
DEFAULT_PERSIST_ROOT = PROJECT_ROOT / ".chunking_selection_bge_m3"

EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 5
FETCH_K = 15
DISTANCE = "l2"


def release_chroma_handles() -> None:
    """Windows에서 Chroma가 잡고 있는 인덱스 파일 핸들을 해제한다."""
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except (ImportError, AttributeError):
        pass
    gc.collect()


def remove_persist_root(path: Path, attempts: int = 6) -> bool:
    """Chroma 디렉터리를 재시도하여 삭제하고 실패해도 예외를 전파하지 않는다."""
    if not path.exists():
        return True

    for attempt in range(1, attempts + 1):
        release_chroma_handles()
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            if attempt < attempts:
                time.sleep(0.5 * attempt)

    print(
        f"경고: 임시 Chroma 인덱스를 삭제하지 못했습니다: {path}\n"
        "평가 결과 저장에는 영향이 없으며, Python 종료 후 직접 삭제할 수 있습니다."
    )
    return False


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """100문항 검색 평가셋을 읽고 필수 필드를 검증한다."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("평가 파일의 최상위 값은 비어 있지 않은 JSON 배열이어야 합니다.")

    normalized = []
    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"평가 항목 #{index}은 JSON 객체여야 합니다.")
        query = str(case.get("query", "")).strip()
        relevant_slides = case.get("relevant_slides")
        if not query:
            raise ValueError(f"평가 항목 #{index}의 query가 비어 있습니다.")
        if not isinstance(relevant_slides, list) or not relevant_slides:
            raise ValueError(
                f"평가 항목 #{index}의 relevant_slides가 비어 있습니다."
            )
        normalized.append(
            {
                "eval_id": str(case.get("eval_id", f"eval-{index:03d}")),
                "query": query,
                "relevant_slides": {int(value) for value in relevant_slides},
            }
        )
    return normalized


def chunk_statistics(chunks: list[dict[str, Any]]) -> dict[str, float | int]:
    lengths = [len(chunk["text"]) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "avg_chars": mean(lengths) if lengths else 0.0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
    }


def evaluate_strategy(
    strategy: str,
    chunks: list[dict[str, Any]],
    eval_set: list[dict[str, Any]],
    embedder: ProductionEmbedder,
    persist_root: Path,
) -> dict[str, Any]:
    """한 청킹 방법을 고정된 BGE-M3 검색 조건으로 평가한다."""
    collection_dir = persist_root / strategy
    if collection_dir.exists() and not remove_persist_root(collection_dir):
        raise PermissionError(
            f"기존 Chroma 인덱스를 삭제할 수 없습니다: {collection_dir}"
        )

    collection = build_index(
        chunks,
        embedder,
        persist_dir=str(collection_dir),
        collection_name=f"bge_m3_{strategy}",
        space=DISTANCE,
    )

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    diversities: list[float] = []

    for case in eval_set:
        retrieved = retrieve_diverse(
            collection,
            embedder,
            case["query"],
            k=TOP_K,
            fetch_k=FETCH_K,
        )
        retrieved_slides = [int(item["slide_no"]) for item in retrieved]
        relevant_slides = case["relevant_slides"]
        recalls.append(recall_at_k(retrieved_slides, relevant_slides))
        reciprocal_ranks.append(mrr(retrieved_slides, relevant_slides))
        precisions.append(context_precision(retrieved_slides, relevant_slides))
        diversities.append(
            len(set(retrieved_slides)) / len(retrieved_slides)
            if retrieved_slides
            else 0.0
        )

    summary = {
        "strategy": strategy,
        "config": CHUNKING_CONFIGS[strategy],
        "evaluation_questions": len(eval_set),
        **chunk_statistics(chunks),
        f"recall@{TOP_K}": mean(recalls),
        "mrr": mean(reciprocal_ranks),
        f"context_precision@{TOP_K}": mean(precisions),
        f"diversity@{TOP_K}": mean(diversities),
    }
    del collection
    return summary


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MRR을 우선하고 Recall·Precision·청크 수로 동률을 처리한다."""
    return sorted(
        results,
        key=lambda row: (
            row["mrr"],
            row[f"recall@{TOP_K}"],
            row[f"context_precision@{TOP_K}"],
            -row["chunks"],
            row["strategy"],
        ),
        reverse=True,
    )


def print_summary(ranking: list[dict[str, Any]]) -> None:
    header = (
        f"{'rank':>4s} {'strategy':14s} {'chunks':>7s} "
        f"{f'recall@{TOP_K}':>10s} {'MRR':>8s} "
        f"{f'precision@{TOP_K}':>12s} {f'diversity@{TOP_K}':>12s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for rank, row in enumerate(ranking, start=1):
        selected = " *" if rank == 1 else ""
        print(
            f"{rank:4d} {row['strategy']:14s} {row['chunks']:7d} "
            f"{row[f'recall@{TOP_K}']:10.4f} {row['mrr']:8.4f} "
            f"{row[f'context_precision@{TOP_K}']:12.4f} "
            f"{row[f'diversity@{TOP_K}']:12.4f}{selected}"
        )
    print(f"\n최종 선별 청킹 방법: {ranking[0]['strategy']}")
    print("* MRR 우선, Recall/Precision/적은 청크 수 순으로 동률 처리")


def write_csv(ranking: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "selected",
        "strategy",
        "chunk_size",
        "overlap",
        "chunks",
        "avg_chars",
        f"recall@{TOP_K}",
        "mrr",
        f"context_precision@{TOP_K}",
        f"diversity@{TOP_K}",
        "evaluation_questions",
        "embedding_model",
        "top_k",
        "fetch_k",
        "distance",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranking, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "selected": rank == 1,
                    "strategy": row["strategy"],
                    "chunk_size": row["config"]["chunk_size"],
                    "overlap": row["config"]["overlap"],
                    "chunks": row["chunks"],
                    "avg_chars": round(row["avg_chars"], 3),
                    f"recall@{TOP_K}": round(row[f"recall@{TOP_K}"], 6),
                    "mrr": round(row["mrr"], 6),
                    f"context_precision@{TOP_K}": round(
                        row[f"context_precision@{TOP_K}"], 6
                    ),
                    f"diversity@{TOP_K}": round(
                        row[f"diversity@{TOP_K}"], 6
                    ),
                    "evaluation_questions": row["evaluation_questions"],
                    "embedding_model": EMBEDDING_MODEL,
                    "top_k": TOP_K,
                    "fetch_k": FETCH_K,
                    "distance": DISTANCE,
                }
            )
    return path


def write_html(ranking: list[dict[str, Any]], output_path: str | Path) -> Path:
    """이미지 없이 표와 CSS 막대로 비교 가능한 독립 HTML을 만든다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    winner = ranking[0]["strategy"]
    table_rows = []
    for rank, row in enumerate(ranking, start=1):
        selected = "selected" if rank == 1 else ""
        marker = "최종 선택" if rank == 1 else ""
        mrr_value = float(row["mrr"])
        table_rows.append(
            f"""
            <tr class="{selected}">
              <td>{rank}</td>
              <td><strong>{html.escape(row["strategy"])}</strong> {marker}</td>
              <td>{row["config"]["chunk_size"]}</td>
              <td>{row["config"]["overlap"]}</td>
              <td>{row["chunks"]}</td>
              <td>{row[f"recall@{TOP_K}"]:.4f}</td>
              <td>
                <span class="bar" style="--score:{mrr_value:.6f}"></span>
                {mrr_value:.4f}
              </td>
              <td>{row[f"context_precision@{TOP_K}"]:.4f}</td>
              <td>{row[f"diversity@{TOP_K}"]:.4f}</td>
            </tr>"""
        )

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BGE-M3 청킹 방법 최종 선별</title>
  <style>
    body {{ margin: 0; padding: 32px; font-family: "Malgun Gothic", sans-serif;
      color: #172033; background: #f4f7fb; }}
    main {{ max-width: 1100px; margin: auto; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ color: #5d6b82; margin-bottom: 24px; }}
    .winner {{ padding: 16px 18px; margin-bottom: 20px; border-radius: 12px;
      background: #e8f2ff; color: #123b80; font-size: 1.1rem; }}
    .table-wrap {{ overflow-x: auto; background: white; border-radius: 14px;
      box-shadow: 0 8px 24px rgba(23,32,51,.07); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{ padding: 13px 12px; text-align: right;
      border-bottom: 1px solid #e3e8f0; }}
    th {{ background: #eef2f7; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tr.selected {{ background: #f0f7ff; }}
    .bar {{ display: inline-block; width: calc(var(--score) * 110px);
      height: 10px; margin-right: 8px; border-radius: 999px;
      background: #2563eb; vertical-align: middle; }}
    .rule {{ margin-top: 16px; color: #5d6b82; }}
  </style>
</head>
<body>
  <main>
    <h1>BGE-M3 청킹 방법 최종 선별</h1>
    <p class="meta">{EMBEDDING_MODEL} · 정규화 · 접두사 미사용 ·
      L2 · top_k={TOP_K} · fetch_k={FETCH_K} ·
      평가질문 {ranking[0]["evaluation_questions"]}개</p>
    <div class="winner">최종 선별: <strong>{html.escape(winner)}</strong></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>순위</th><th>청킹 방법</th><th>크기</th><th>중첩</th>
          <th>청크 수</th><th>Recall@{TOP_K}</th><th>MRR</th>
          <th>Precision@{TOP_K}</th><th>Diversity@{TOP_K}</th></tr>
        </thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table>
    </div>
    <p class="rule">선별 기준: MRR 우선, Recall, Precision, 적은 청크 수 순.</p>
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    pptx_path = Path(args.pptx)
    eval_set = load_eval_set(args.eval_file)
    print(f"PPT 추출: {pptx_path}")
    flat_documents = extract_slide_texts(pptx_path)
    structured_documents = extract_slide_structures(pptx_path)
    chunk_sets = build_all_chunk_sets(flat_documents, structured_documents)

    print(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
    embedder = ProductionEmbedder(EMBEDDING_MODEL, use_prefix=False)
    persist_root = Path(args.persist_root)
    results = []
    for strategy, chunks in chunk_sets.items():
        print(f"평가 중: {strategy} ({len(chunks)} chunks)")
        results.append(
            evaluate_strategy(
                strategy,
                chunks,
                eval_set,
                embedder,
                persist_root,
            )
        )

    ranking = rank_results(results)
    print_summary(ranking)
    csv_path = write_csv(ranking, args.csv_output)
    html_path = write_html(ranking, args.html_output)
    print(f"CSV 저장: {csv_path}")
    print(f"HTML 저장: {html_path}")

    if not args.keep_index:
        # 결과 저장이 끝난 뒤 정리한다. Windows 파일 잠금으로 삭제가
        # 실패하더라도 이미 저장된 평가 결과에는 영향이 없다.
        remove_persist_root(persist_root)
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx", default=str(DEFAULT_PPTX))
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_OUTPUT))
    parser.add_argument("--html-output", default=str(DEFAULT_HTML_OUTPUT))
    parser.add_argument("--persist-root", default=str(DEFAULT_PERSIST_ROOT))
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="평가 후 임시 Chroma 인덱스를 삭제하지 않는다.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
