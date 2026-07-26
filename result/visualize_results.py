"""청킹 평가점수와 생성 퀴즈를 하나의 HTML 리포트로 시각화한다.

사용법:
    python result/visualize_results.py
    python result/visualize_results.py --chunking-result result/chunking_results.json
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


QUIZ_FIELDS = {
    "quiz_id",
    "type",
    "question",
    "choices",
    "answer",
    "explanation",
    "evidence",
    "source",
}
COLORS = {
    "recall": "#3157d5",
    "mrr": "#8b5cf6",
    "precision": "#08a88a",
    "chunks": "#f59e0b",
    "chars": "#ef6c57",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_chunking_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("results", [])
    if not isinstance(data, list):
        raise ValueError("청킹 평가 결과는 JSON 배열이어야 합니다.")
    return [row for row in data if isinstance(row, dict) and "strategy" in row]


def load_quizzes(directory: Path) -> list[tuple[str, dict[str, Any]]]:
    quizzes: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [data]
        for index, record in enumerate(records, start=1):
            if isinstance(record, dict) and QUIZ_FIELDS.issubset(record):
                label = path.name if len(records) == 1 else f"{path.name} #{index}"
                quizzes.append((label, record))
    return quizzes


def metric_keys(results: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    keys = {key for row in results for key in row}
    recall = next((key for key in sorted(keys) if key.startswith("recall@")), None)
    precision = next(
        (key for key in sorted(keys) if key.startswith("context_precision@")),
        None,
    )
    return recall, precision


def score_bar(label: str, value: float, color: str) -> str:
    percentage = max(0.0, min(1.0, value)) * 100
    return f"""
      <div class="metric">
        <div class="metric-label"><span>{esc(label)}</span><strong>{value:.3f}</strong></div>
        <div class="track"><div class="fill" style="width:{percentage:.2f}%;background:{color}"></div></div>
      </div>
    """


def score_charts(results: list[dict[str, Any]]) -> str:
    recall_key, precision_key = metric_keys(results)
    if not results or not recall_key or not precision_key:
        return """
        <div class="notice">
          평가점수 결과가 없습니다. 먼저 compare_chunking.py를 실행해
          result/chunking_results.json을 생성하세요.
        </div>
        """

    cards = []
    for row in results:
        evaluation_count = int(row.get("evaluation_questions", len(row.get("cases", []))))
        cards.append(
            f"""
            <article class="score-card">
              <div class="score-head">
                <h3>{esc(row["strategy"])}</h3>
                <span>{evaluation_count:,}문항 · {int(row.get("chunks", 0)):,} chunks</span>
              </div>
              {score_bar(recall_key, float(row.get(recall_key, 0)), COLORS["recall"])}
              {score_bar("MRR", float(row.get("mrr", 0)), COLORS["mrr"])}
              {score_bar(precision_key, float(row.get(precision_key, 0)), COLORS["precision"])}
            </article>
            """
        )
    return f'<div class="score-grid">{"".join(cards)}</div>'


def case_rank(case: dict[str, Any]) -> int | None:
    """Return the first rank containing a relevant slide."""
    relevant = set(case.get("relevant_slides", []))
    for rank, slide in enumerate(case.get("retrieved_slides", []), start=1):
        if slide in relevant:
            return rank
    return None


def rank_distribution(results: list[dict[str, Any]]) -> str:
    """Visualize how often the correct slide appeared at each retrieval rank."""
    if not results:
        return ""
    cards = []
    segment_colors = ["#3157d5", "#8b5cf6", "#08a88a", "#d6dce6"]
    labels = ["Top-1", "Top-2", "Top-3", "미검색"]
    for row in results:
        counts = [0, 0, 0, 0]
        cases = row.get("cases", [])
        for case in cases:
            rank = case_rank(case)
            counts[rank - 1 if rank in (1, 2, 3) else 3] += 1
        total = len(cases) or 1
        segments = "".join(
            (
                f'<div class="rank-segment" title="{labels[index]}: {count}개" '
                f'style="width:{count / total * 100:.2f}%;'
                f'background:{segment_colors[index]}"></div>'
            )
            for index, count in enumerate(counts)
            if count
        )
        legend = "".join(
            f'<span><i style="background:{segment_colors[index]}"></i>'
            f"{labels[index]} <b>{count}</b></span>"
            for index, count in enumerate(counts)
        )
        cards.append(
            f"""
            <article class="rank-card">
              <h3>{esc(row["strategy"])}</h3>
              <div class="rank-track">{segments}</div>
              <div class="rank-legend">{legend}</div>
            </article>
            """
        )
    return f'<div class="rank-grid">{"".join(cards)}</div>'


def evaluation_detail_table(results: list[dict[str, Any]]) -> str:
    """Create a collapsible per-question comparison table."""
    if not results or not results[0].get("cases"):
        return ""
    strategies = [str(row["strategy"]) for row in results]
    case_maps = {
        str(row["strategy"]): {
            str(case.get("eval_id", index)): case
            for index, case in enumerate(row.get("cases", []), start=1)
        }
        for row in results
    }
    base_cases = results[0]["cases"]
    headers = "".join(f"<th>{esc(strategy)}</th>" for strategy in strategies)
    rows = []
    for index, base in enumerate(base_cases, start=1):
        eval_id = str(base.get("eval_id", f"eval-{index:03d}"))
        score_cells = []
        for strategy in strategies:
            case = case_maps[strategy].get(eval_id, {})
            rank = case_rank(case)
            rank_text = f"Top-{rank}" if rank is not None else "미검색"
            css_class = f"rank-{rank}" if rank is not None else "rank-miss"
            slides = ", ".join(str(value) for value in case.get("retrieved_slides", []))
            score_cells.append(
                f'<td><span class="rank-pill {css_class}">{rank_text}</span>'
                f'<small class="slides">[{esc(slides)}]</small></td>'
            )
        rows.append(
            "<tr>"
            f"<td>{esc(eval_id)}</td>"
            f"<td>{esc(base.get('category', '미분류'))}</td>"
            f"<td class=\"query-cell\">{esc(base.get('query', ''))}</td>"
            f"<td>{esc(', '.join(str(v) for v in base.get('relevant_slides', [])))}</td>"
            f"{''.join(score_cells)}"
            "</tr>"
        )
    return f"""
    <details class="detail-panel">
      <summary>평가 질문 {len(base_cases)}개 상세 검색 결과 보기</summary>
      <div class="table-wrap detail-table">
        <table>
          <thead><tr>
            <th>ID</th><th>분류</th><th>평가 질문</th><th>정답 슬라이드</th>{headers}
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </details>
    """


def stats_chart(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    max_chunks = max(float(row.get("chunks", 0)) for row in results) or 1
    max_chars = max(float(row.get("avg_chars", 0)) for row in results) or 1
    rows = []
    for row in results:
        chunks = float(row.get("chunks", 0))
        chars = float(row.get("avg_chars", 0))
        rows.append(
            f"""
            <div class="stat-row">
              <strong>{esc(row["strategy"])}</strong>
              <div>
                <div class="mini-label"><span>청크 수</span><b>{int(chunks):,}</b></div>
                <div class="track"><div class="fill" style="width:{chunks / max_chunks * 100:.2f}%;background:{COLORS["chunks"]}"></div></div>
              </div>
              <div>
                <div class="mini-label"><span>평균 글자 수</span><b>{chars:.1f}</b></div>
                <div class="track"><div class="fill" style="width:{chars / max_chars * 100:.2f}%;background:{COLORS["chars"]}"></div></div>
              </div>
            </div>
            """
        )
    return f'<div class="stats-chart">{"".join(rows)}</div>'


def score_table(results: list[dict[str, Any]]) -> str:
    recall_key, precision_key = metric_keys(results)
    if not results:
        return ""
    headers = [
        "청킹 방법",
        "청크 수",
        "평균 길이",
        recall_key or "Recall",
        "MRR",
        precision_key or "Context Precision",
    ]
    rows = []
    for row in results:
        def value(key: str | None) -> str:
            return f"{float(row.get(key, 0)):.3f}" if key else "-"

        rows.append(
            "<tr>"
            f"<td><strong>{esc(row['strategy'])}</strong></td>"
            f"<td>{int(row.get('chunks', 0)):,}</td>"
            f"<td>{float(row.get('avg_chars', 0)):.1f}</td>"
            f"<td>{value(recall_key)}</td>"
            f"<td>{value('mrr')}</td>"
            f"<td>{value(precision_key)}</td>"
            "</tr>"
        )
    header_html = "".join(f"<th>{esc(header)}</th>" for header in headers)
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def answer_text(quiz: dict[str, Any]) -> str:
    answer = quiz.get("answer")
    choices = quiz.get("choices", [])
    if (
        isinstance(answer, int)
        and not isinstance(answer, bool)
        and isinstance(choices, list)
        and 1 <= answer <= len(choices)
    ):
        return f"{answer}번 · {choices[answer - 1]}"
    return str(answer)


def choice_list(quiz: dict[str, Any]) -> str:
    choices = quiz.get("choices", [])
    answer = quiz.get("answer")
    if not choices:
        return '<p class="muted">선택지 없음</p>'
    items = []
    for index, choice in enumerate(choices, start=1):
        correct = isinstance(answer, int) and not isinstance(answer, bool) and answer == index
        css_class = "choice correct" if correct else "choice"
        badge = '<span class="badge">정답</span>' if correct else ""
        items.append(
            f'<li class="{css_class}"><span class="number">{index}</span>'
            f"<span>{esc(choice)}</span>{badge}</li>"
        )
    return f'<ol class="choices">{"".join(items)}</ol>'


def quiz_card(filename: str, quiz: dict[str, Any]) -> str:
    source = quiz.get("source") if isinstance(quiz.get("source"), dict) else {}
    return f"""
    <article class="quiz-card">
      <div class="quiz-head">
        <div><small>{esc(filename)}</small><h3>{esc(quiz.get("quiz_id", "-"))}</h3></div>
        <span class="type">{esc(quiz.get("type", "-"))}</span>
      </div>
      <p class="question">{esc(quiz.get("question", ""))}</p>
      {choice_list(quiz)}
      <dl>
        <dt>정답</dt><dd class="answer">{esc(answer_text(quiz))}</dd>
        <dt>해설</dt><dd>{esc(quiz.get("explanation", ""))}</dd>
        <dt>근거</dt><dd>{esc(quiz.get("evidence", ""))}</dd>
      </dl>
      <div class="source">
        <span>{esc(source.get("file", "-"))}</span>
        <span>슬라이드 {esc(source.get("slide", "-"))}</span>
        <code>{esc(source.get("chunk_id", "-"))}</code>
      </div>
    </article>
    """


def quiz_section(quizzes: list[tuple[str, dict[str, Any]]]) -> str:
    if not quizzes:
        return '<div class="notice">시각화할 퀴즈 JSON이 없습니다.</div>'
    cards = "".join(quiz_card(filename, quiz) for filename, quiz in quizzes)
    return f'<div class="quiz-grid">{cards}</div>'


def build_html(
    results: list[dict[str, Any]],
    quizzes: list[tuple[str, dict[str, Any]]],
) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG 청킹 평가 및 퀴즈 결과</title>
  <style>
    :root {{
      --ink:#18212f; --muted:#667085; --line:#dce3ed; --surface:#fff;
      --canvas:#f3f6fa; --brand:#3157d5; --correct:#18794e; --correct-bg:#eaf8f0;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); background:var(--canvas);
      font-family:Pretendard,"Noto Sans KR","Malgun Gothic",sans-serif;
    }}
    main {{ width:min(1440px,94vw); margin:46px auto 80px; }}
    header {{ margin-bottom:38px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(28px,3vw,44px); }}
    h2 {{ margin:0 0 8px; font-size:26px; }}
    h3 {{ margin:0; }}
    .subtitle,.muted,small {{ color:var(--muted); }}
    section {{ margin-top:42px; }}
    .section-head {{ margin-bottom:20px; }}
    .section-head p {{ color:var(--muted); margin:6px 0 0; }}
    .score-grid {{
      display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); gap:16px;
    }}
    .score-card,.quiz-card,.stats-chart,.notice {{
      background:var(--surface); border:1px solid var(--line); border-radius:17px;
      box-shadow:0 7px 24px rgba(33,45,71,.055);
    }}
    .score-card {{ padding:20px; }}
    .score-head {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:20px; }}
    .score-head span {{ color:var(--muted); font-size:12px; }}
    .metric {{ margin-top:15px; }}
    .metric-label,.mini-label {{ display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px; }}
    .metric-label span,.mini-label span {{ color:var(--muted); }}
    .track {{ height:9px; border-radius:999px; background:#edf0f5; overflow:hidden; }}
    .fill {{ height:100%; border-radius:inherit; }}
    .rank-grid {{
      display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
      gap:16px; margin-top:18px;
    }}
    .rank-card {{
      padding:20px; background:var(--surface); border:1px solid var(--line);
      border-radius:17px;
    }}
    .rank-track {{
      display:flex; height:18px; overflow:hidden; border-radius:999px;
      background:#edf0f5; margin:17px 0 13px;
    }}
    .rank-segment {{ height:100%; }}
    .rank-legend {{ display:flex; flex-wrap:wrap; gap:8px 14px; color:var(--muted); font-size:12px; }}
    .rank-legend span {{ display:flex; align-items:center; gap:5px; }}
    .rank-legend i {{ width:8px; height:8px; border-radius:50%; }}
    .rank-legend b {{ color:var(--ink); }}
    .stats-chart {{ padding:20px; margin-top:18px; }}
    .stat-row {{
      display:grid; grid-template-columns:140px 1fr 1fr; align-items:center;
      gap:22px; padding:14px 0; border-bottom:1px solid var(--line);
    }}
    .stat-row:last-child {{ border-bottom:0; }}
    .table-wrap {{
      overflow-x:auto; margin-top:18px; background:var(--surface);
      border:1px solid var(--line); border-radius:14px;
    }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ padding:13px 16px; border-bottom:1px solid var(--line); text-align:left; }}
    th {{ color:var(--muted); background:#f9fafc; white-space:nowrap; }}
    tr:last-child td {{ border-bottom:0; }}
    .detail-panel {{
      margin-top:18px; background:var(--surface); border:1px solid var(--line);
      border-radius:14px; overflow:hidden;
    }}
    .detail-panel summary {{ padding:16px 18px; cursor:pointer; font-weight:750; }}
    .detail-panel .table-wrap {{ margin:0; border:0; border-top:1px solid var(--line); border-radius:0; }}
    .detail-table table {{ min-width:1120px; }}
    .query-cell {{ min-width:320px; }}
    .rank-pill {{ display:inline-block; padding:4px 7px; border-radius:999px; font-size:11px; font-weight:800; }}
    .rank-1 {{ color:#2347bd; background:#eaf0ff; }}
    .rank-2 {{ color:#7042c2; background:#f1eafe; }}
    .rank-3 {{ color:#087d69; background:#e5f8f3; }}
    .rank-miss {{ color:#687386; background:#edf0f4; }}
    .slides {{ display:block; margin-top:5px; white-space:nowrap; }}
    .quiz-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:20px; }}
    .quiz-card {{ padding:24px; }}
    .quiz-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
    .quiz-head h3 {{ margin-top:4px; }}
    .type {{
      color:var(--brand); background:#edf1ff; border-radius:999px;
      padding:7px 10px; font-size:12px; font-weight:700;
    }}
    .question {{ margin:23px 0 17px; font-size:20px; font-weight:750; line-height:1.45; }}
    .choices {{ list-style:none; padding:0; margin:0 0 22px; display:grid; gap:9px; }}
    .choice {{
      display:flex; align-items:center; gap:10px; padding:11px 12px;
      border:1px solid var(--line); border-radius:10px;
    }}
    .choice.correct {{ border-color:#83c5a2; background:var(--correct-bg); color:var(--correct); }}
    .number {{
      display:grid; place-items:center; min-width:25px; height:25px;
      border-radius:50%; color:#fff; background:#8190a8; font-size:12px; font-weight:700;
    }}
    .correct .number {{ background:var(--correct); }}
    .badge {{ margin-left:auto; font-size:11px; font-weight:800; }}
    dl {{ display:grid; grid-template-columns:52px 1fr; gap:11px 12px; margin:0; line-height:1.55; }}
    dt {{ color:var(--muted); font-size:13px; font-weight:700; }}
    dd {{ margin:0; font-size:14px; }}
    dd.answer {{ color:var(--correct); font-weight:800; }}
    .source {{
      display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; padding-top:16px;
      border-top:1px solid var(--line); color:var(--muted); font-size:12px;
    }}
    .source span,.source code {{ background:#f4f6f9; border-radius:7px; padding:6px 8px; }}
    code {{ font-family:Consolas,monospace; font-size:12px; }}
    .notice {{ padding:20px; color:var(--muted); }}
    @media (max-width:720px) {{
      .stat-row {{ grid-template-columns:1fr; gap:10px; }}
      .quiz-grid {{ grid-template-columns:1fr; }}
      main {{ margin-top:28px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>RAG 실험 결과 대시보드</h1>
    <p class="subtitle">청킹 전략별 검색 품질과 생성된 퀴즈를 함께 비교합니다.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>1. 청킹 방법별 평가점수</h2>
      <p>동일한 평가 질문과 검색 조건에서 Recall, MRR, Context Precision을 비교합니다.</p>
    </div>
    {score_charts(results)}
    {score_table(results)}
    <div class="section-head rank-heading">
      <h2>정답 슬라이드 검색 순위 분포</h2>
      <p>각 평가 질문의 정답 슬라이드가 Top-1, Top-2, Top-3에 검색된 횟수와 미검색 횟수입니다.</p>
    </div>
    {rank_distribution(results)}
    {evaluation_detail_table(results)}
  </section>

  <section>
    <div class="section-head">
      <h2>2. 청크 구성 비교</h2>
      <p>전략별 청크 수와 평균 길이를 각 지표의 최댓값 기준으로 표시합니다.</p>
    </div>
    {stats_chart(results)}
  </section>

  <section>
    <div class="section-head">
      <h2>3. 생성 퀴즈</h2>
      <p>문제, 선택지, 정답, 해설, 근거와 원문 출처를 확인합니다.</p>
    </div>
    {quiz_section(quizzes)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunking-result",
        type=Path,
        default=script_dir / "chunking_results.json",
        help="compare_chunking.py가 생성한 평가 결과 JSON",
    )
    parser.add_argument(
        "--quiz-dir",
        type=Path,
        default=script_dir,
        help="생성된 퀴즈 JSON 디렉터리",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "rag_result_dashboard.html",
        help="생성할 HTML 파일",
    )
    args = parser.parse_args()

    results = load_chunking_results(args.chunking_result)
    quizzes = load_quizzes(args.quiz_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(results, quizzes), encoding="utf-8")
    print(
        f"대시보드 생성 완료: {args.output} "
        f"(청킹 전략 {len(results)}개, 퀴즈 {len(quizzes)}개)"
    )


if __name__ == "__main__":
    main()
