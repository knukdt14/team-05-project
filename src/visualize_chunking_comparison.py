"""all_chunking_quizzes.json을 전략별 비교 HTML로 변환한다."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from settings import PROJECT_ROOT


DEFAULT_INPUT = (
    PROJECT_ROOT / "results" / "chunking_comparison" / "all_chunking_quizzes.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "chunking_comparison" / "all_chunking_quizzes.html"
)


def load_comparison(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
        raise ValueError("비교 JSON은 experiments 배열을 가진 객체여야 합니다.")
    experiments = [item for item in data["experiments"] if isinstance(item, dict)]
    if not experiments:
        raise ValueError("비교 JSON에 실험 결과가 없습니다.")
    data["experiments"] = experiments
    return data


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _answer_text(quiz: dict[str, Any]) -> str:
    answer = quiz.get("answer", "")
    choices = quiz.get("choices") or []
    if isinstance(answer, int) and 1 <= answer <= len(choices):
        return f"{answer}번 — {choices[answer - 1]}"
    return str(answer)


def _strategy_name(experiment: dict[str, Any]) -> str:
    config = experiment.get("chunking") or {}
    return (
        f"{config.get('strategy', 'unknown')} "
        f"(size={config.get('chunk_size', '-')}, overlap={config.get('overlap', '-')})"
    )


def _choices_markup(quiz: dict[str, Any]) -> str:
    choices = quiz.get("choices") or []
    if not choices:
        return '<p class="muted">보기가 없습니다.</p>'
    return "<ol>" + "".join(
        f"<li>{_escape(choice)}</li>" for choice in choices
    ) + "</ol>"


def _used_chunks_markup(quiz: dict[str, Any]) -> str:
    retrieved = quiz.get("retrieved_chunks") or []
    sources = quiz.get("sources") or []
    source_ids = {
        str(source.get("chunk_id"))
        for source in sources
        if isinstance(source, dict) and source.get("chunk_id")
    }
    used = [
        chunk for chunk in retrieved
        if str(chunk.get("chunk_id")) in source_ids
    ]
    if not used:
        used = retrieved
    if not used:
        return '<p class="muted">연결된 검색 청크가 없습니다.</p>'

    label = "생성에 사용된 청크" if source_ids else "검색된 청크"
    cards = []
    for chunk in used:
        cards.append(
            '<div class="used-chunk">'
            f'<div class="chunk-meta">{_escape(chunk.get("chunk_id", ""))} · '
            f'슬라이드 {_escape(chunk.get("slide_no", ""))} · '
            f'distance={_escape(chunk.get("distance", ""))}</div>'
            f'<pre>{_escape(chunk.get("text", ""))}</pre>'
            "</div>"
        )
    return f'<div class="used-chunks"><b>{label}</b>{"".join(cards)}</div>'


def _quiz_markup(quiz: dict[str, Any], index: int) -> str:
    return f"""
    <article class="quiz-card">
      <div class="meta"><span class="badge">문제 {index}</span>
        <span>{_escape(quiz.get("topic", "주제 없음"))}</span>
        <code>{_escape(quiz.get("quiz_id", ""))}</code></div>
      <h3>{_escape(quiz.get("question", "질문 없음"))}</h3>
      {_choices_markup(quiz)}
      <details>
        <summary>정답·해설 보기</summary>
        <div class="answer"><b>정답</b> {_escape(_answer_text(quiz))}</div>
        <p><b>해설</b><br>{_escape(quiz.get("explanation", ""))}</p>
        <p class="evidence"><b>근거</b><br>{_escape(quiz.get("evidence", ""))}</p>
      </details>
      {_used_chunks_markup(quiz)}
    </article>
    """


def _overview_markup(data: dict[str, Any]) -> str:
    experiments = data["experiments"]
    topics = data.get("topics") or sorted(
        {
            quiz.get("topic", "")
            for experiment in experiments
            for quiz in experiment.get("quizzes", [])
            if quiz.get("topic")
        }
    )
    maps = [
        {quiz.get("topic"): quiz for quiz in experiment.get("quizzes", [])}
        for experiment in experiments
    ]
    headers = "".join(f"<th>{_escape(_strategy_name(e))}</th>" for e in experiments)
    rows = []
    for topic in topics:
        cells = []
        for quiz_map in maps:
            quiz = quiz_map.get(topic)
            if quiz:
                cells.append(
                    '<td class="ok">생성됨<br>'
                    f'<span class="muted">{_escape(quiz.get("quiz_id", ""))}</span></td>'
                )
            else:
                cells.append('<td class="failed">실패/없음</td>')
        rows.append(f"<tr><td>{_escape(topic)}</td>{''.join(cells)}</tr>")
    body = "".join(rows) or '<tr><td colspan="2">주제가 없습니다.</td></tr>'
    metric_cards = []
    for experiment in experiments:
        mean = experiment.get("retrieval_metrics_mean") or {}
        retrieval_text = (
            f'R@K {float(mean.get("recall_at_k", 0)):.3f} · '
            f'P@K {float(mean.get("precision_at_k", 0)):.3f} · '
            f'MRR {float(mean.get("mrr", 0)):.3f}'
            if mean else "검색 정답 라벨 없음"
        )
        metric_cards.append(
            f'<div class="metric"><b>{len(experiment.get("quizzes", []))}</b>'
            f'{_escape(_strategy_name(experiment))}<br><span class="muted">'
            f'실패 {len(experiment.get("failures", []))}개 · {retrieval_text}</span></div>'
        )
    return f"""
    <section class="panel-section active" id="overview">
      <div class="panel">
        <h2>전체 비교 요약</h2>
        <div class="metrics">
          {"".join(metric_cards)}
        </div>
      </div>
      <div class="panel table-wrap">
        <h2>동일 주제 생성 여부</h2>
        <table><thead><tr><th>주제</th>{headers}</tr></thead>
        <tbody>{body}</tbody></table>
      </div>
    </section>
    """


def _experiment_markup(experiment: dict[str, Any], index: int) -> str:
    quizzes = experiment.get("quizzes") or []
    topics: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for quiz_index, quiz in enumerate(quizzes, start=1):
        topic = str(quiz.get("topic") or "주제 없음")
        topics.setdefault(topic, []).append((quiz_index, quiz))
    topic_groups = []
    for topic, topic_quizzes in topics.items():
        cards = "".join(
            _quiz_markup(quiz, quiz_index)
            for quiz_index, quiz in topic_quizzes
        )
        topic_groups.append(
            f'<section class="topic-group"><h3>주제: {_escape(topic)}</h3>'
            f'<div class="quiz-list">{cards}</div></section>'
        )
    if not topic_groups:
        topic_groups = ['<div class="panel">생성된 질문이 없습니다.</div>']
    mean = experiment.get("retrieval_metrics_mean") or {}
    metric_text = (
        f' · Recall@K {float(mean.get("recall_at_k", 0)):.3f}'
        f' · Precision@K {float(mean.get("precision_at_k", 0)):.3f}'
        f' · MRR {float(mean.get("mrr", 0)):.3f}'
        if mean else ""
    )
    return f"""
    <section class="panel-section" id="experiment-{index}">
      <div class="panel">
        <h2>{_escape(_strategy_name(experiment))}</h2>
        <p class="muted">퀴즈 {len(quizzes)}개 · 실패 {len(experiment.get("failures") or [])}개{metric_text}</p>
      </div>
      <div class="topic-list">{"".join(topic_groups)}</div>
    </section>
    """


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root { --bg:#f4f7fb; --card:#fff; --ink:#172033; --muted:#667085;
            --line:#d8e0ec; --brand:#2457d6; --soft:#e9efff;
            --ok:#eaf8ef; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink);
      font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif; line-height:1.55; }
    main { width:min(1180px,calc(100% - 32px)); margin:32px auto 70px; }
    h1 { margin:0 0 6px; font-size:clamp(1.5rem,3vw,2.2rem); }
    h2 { margin:0 0 14px; font-size:1.18rem; }
    h3 { margin:12px 0; font-size:1.08rem; }
    h4 { margin-bottom:7px; }
    .muted { color:var(--muted); }
    .header { margin-bottom:24px; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; margin:20px 0; }
    .tab { border:1px solid var(--line); background:var(--card); color:var(--ink);
      border-radius:999px; padding:9px 15px; text-decoration:none; }
    .tab:hover, .tab.active { background:var(--brand); border-color:var(--brand); color:#fff; }
    .panel-section { display:none; }
    .panel-section.active { display:block; }
    .panel { background:var(--card); border:1px solid var(--line); border-radius:16px;
      padding:20px; margin-bottom:20px; box-shadow:0 5px 18px #1720330c; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }
    .metric { background:var(--soft); border-radius:12px; padding:12px 14px; }
    .metric b { display:block; font-size:1.3rem; color:var(--brand); }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:collapse; min-width:650px; }
    th,td { border-bottom:1px solid var(--line); padding:10px; text-align:left; vertical-align:top; }
    th { color:var(--muted); font-size:.88rem; }
    .ok { color:#137333; font-weight:700; }
    .failed { color:#b54708; font-weight:700; }
    .topic-list { display:grid; gap:28px; }
    .topic-group { background:var(--soft); border-radius:14px; padding:16px; }
    .topic-group > h3 { margin:0 0 14px; color:var(--brand); }
    .quiz-list { display:grid; gap:16px; }
    .quiz-card { background:var(--card); border:1px solid var(--line); border-radius:13px; padding:18px; }
    .used-chunks { margin-top:16px; padding-top:12px; border-top:1px solid var(--line); }
    .used-chunk { background:#f8fafc; border:1px solid var(--line); border-radius:9px; margin:9px 0; padding:11px 12px; }
    .meta { display:flex; gap:8px; align-items:center; flex-wrap:wrap; color:var(--muted); font-size:.88rem; }
    .badge { color:var(--brand); background:var(--soft); border-radius:999px; padding:3px 9px; font-weight:700; }
    code { margin-left:auto; }
    ol { padding-left:28px; }
    details { margin-top:14px; border-top:1px solid var(--line); padding-top:12px; }
    summary { cursor:pointer; color:var(--brand); font-weight:700; }
    .answer { background:var(--ok); padding:10px 12px; border-radius:9px; margin-top:12px; }
    .evidence { white-space:pre-wrap; }
    .chunk { background:#f8fafc; border:1px solid var(--line); border-radius:9px; margin:9px 0; padding:11px 12px; }
    .chunk-meta { color:var(--muted); font-size:.83rem; margin-bottom:5px; }
    pre { white-space:pre-wrap; word-break:break-word; margin:0; font:inherit; font-size:.92rem; }
    @media (max-width:600px) { main { width:calc(100% - 20px); margin-top:20px; } .panel,.quiz-card { padding:15px; } code { width:100%; margin:0; } }
    @media print { body { background:#fff; } .tabs { display:none; } .panel,.quiz-card { box-shadow:none; break-inside:avoid; } }
  </style>
</head>
<body>
  <main>
    <header class="header">
      <h1>청킹 방식별 퀴즈 비교</h1>
      <p class="muted">입력 파일: __INPUT__</p>
      <nav class="tabs">
        <a class="tab active" href="#overview" onclick="showPanel('overview'); return false;">전체 비교</a>
        __TABS__
      </nav>
    </header>
    __CONTENT__
  </main>
  <script>
    function showPanel(id) {
      document.querySelectorAll('.panel-section').forEach(function (section) {
        section.classList.toggle('active', section.id === id);
      });
      document.querySelectorAll('.tab').forEach(function (tab) {
        tab.classList.toggle('active', tab.getAttribute('href') === '#' + id);
      });
      window.location.hash = id;
    }
    window.addEventListener('load', function () {
      var id = window.location.hash.slice(1);
      if (id && document.getElementById(id)) showPanel(id);
    });
  </script>
</body>
</html>
"""


def render_comparison(data: dict[str, Any], *, input_path: str | Path) -> str:
    experiments = data["experiments"]
    tabs = "".join(
        f'<a class="tab" href="#experiment-{index}" '
        f'onclick="showPanel(\'experiment-{index}\'); return false;">'
        f'{_escape(_strategy_name(experiment))}</a>'
        for index, experiment in enumerate(experiments)
    )
    content = _overview_markup(data) + "".join(
        _experiment_markup(experiment, index)
        for index, experiment in enumerate(experiments)
    )
    return (
        HTML_TEMPLATE.replace("__TITLE__", _escape("청킹 방식별 퀴즈 비교"))
        .replace("__INPUT__", _escape(input_path))
        .replace("__TABS__", tabs)
        .replace("__CONTENT__", content)
    )


def visualize_comparison(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    data = load_comparison(input_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_comparison(data, input_path=input_path), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"비교 JSON이 없습니다: {args.input}")
    output = visualize_comparison(args.input, args.output)
    print(f"청킹 비교 시각화 저장: {output}")


if __name__ == "__main__":
    main()
