"""생성된 퀴즈 JSON을 정답 확인이 가능한 HTML 문서로 시각화한다."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from settings import PROJECT_ROOT


DEFAULT_INPUT = PROJECT_ROOT / "results" / "generated_quizzes.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "generated_quizzes.html"


def load_quizzes(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("quizzes")
    if not isinstance(data, list):
        raise ValueError("퀴즈 JSON은 배열이거나 quizzes 배열을 가진 객체여야 합니다.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("퀴즈 배열의 각 항목은 JSON 객체여야 합니다.")
    return data


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _answer_text(quiz: dict[str, Any]) -> str:
    answer = quiz.get("answer", "")
    choices = quiz.get("choices") or []
    if isinstance(answer, int) and 1 <= answer <= len(choices):
        return f"{answer}번 — {choices[answer - 1]}"
    return str(answer)


def _choices_markup(quiz: dict[str, Any]) -> str:
    choices = quiz.get("choices") or []
    if not choices:
        return '<p class="no-choices">보기가 없는 단답형 문제입니다.</p>'
    return "\n".join(
        (
            '<li><span class="choice-number">'
            f"{index}</span><span>{_escape(choice)}</span></li>"
        )
        for index, choice in enumerate(choices, start=1)
    ).join(["<ol class=\"choices\">\n", "\n</ol>"])


def _source_markup(source: Any) -> str:
    if not isinstance(source, dict):
        return ""
    parts = []
    if source.get("file") not in (None, ""):
        parts.append(_escape(source["file"]))
    if source.get("slide") not in (None, ""):
        parts.append(f"슬라이드 {_escape(source['slide'])}")
    if source.get("chunk_id") not in (None, ""):
        parts.append(_escape(source["chunk_id"]))
    return " · ".join(parts)


def render_quizzes(quizzes: list[dict[str, Any]], title: str) -> str:
    type_names = {
        "multiple_choice": "객관식",
        "true_false": "OX",
        "short_answer": "단답형",
    }
    cards = []
    for index, quiz in enumerate(quizzes, start=1):
        quiz_type = str(quiz.get("type", "unknown"))
        source = _source_markup(quiz.get("source"))
        cards.append(
            f"""
      <article class="quiz-card">
        <header>
          <span class="quiz-number">문제 {index}</span>
          <span class="quiz-type">{_escape(type_names.get(quiz_type, quiz_type))}</span>
          <code>{_escape(quiz.get("quiz_id", ""))}</code>
        </header>
        <h2>{_escape(quiz.get("question", "질문 없음"))}</h2>
        {_choices_markup(quiz)}
        <details>
          <summary>정답과 해설 보기</summary>
          <div class="answer"><strong>정답</strong> {_escape(_answer_text(quiz))}</div>
          <p><strong>해설</strong><br>{_escape(quiz.get("explanation", ""))}</p>
          <p><strong>근거</strong><br>{_escape(quiz.get("evidence", ""))}</p>
          <p class="source"><strong>출처</strong> {source or "출처 정보 없음"}</p>
        </details>
      </article>"""
        )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --background: #f4f7fb;
      --card: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d8e0ec;
      --primary: #2457d6;
      --primary-soft: #e9efff;
      --answer: #eaf8ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--background);
      color: var(--text);
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
      line-height: 1.65;
    }}
    main {{ width: min(940px, calc(100% - 32px)); margin: 40px auto; }}
    .page-header {{ margin-bottom: 28px; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(1.65rem, 4vw, 2.25rem); }}
    .summary {{ margin: 0; color: var(--muted); }}
    .quiz-list {{ display: grid; gap: 20px; }}
    .quiz-card {{
      padding: 24px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(23, 32, 51, 0.06);
    }}
    .quiz-card header {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .quiz-number {{ font-weight: 700; color: var(--primary); }}
    .quiz-type {{
      padding: 2px 9px;
      border-radius: 999px;
      background: var(--primary-soft);
      color: var(--primary);
      font-size: 0.85rem;
    }}
    code {{ margin-left: auto; color: var(--muted); }}
    h2 {{ margin: 16px 0; font-size: 1.2rem; }}
    .choices {{ list-style: none; padding: 0; display: grid; gap: 9px; }}
    .choices li {{
      display: flex;
      gap: 10px;
      padding: 11px 13px;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}
    .choice-number {{
      display: inline-grid;
      place-items: center;
      min-width: 25px;
      height: 25px;
      border-radius: 50%;
      background: var(--primary-soft);
      color: var(--primary);
      font-weight: 700;
    }}
    .no-choices {{ color: var(--muted); }}
    details {{ margin-top: 18px; border-top: 1px solid var(--line); padding-top: 14px; }}
    summary {{ cursor: pointer; color: var(--primary); font-weight: 700; }}
    .answer {{ margin-top: 14px; padding: 12px 14px; background: var(--answer); border-radius: 10px; }}
    .source {{ color: var(--muted); word-break: break-all; }}
    @media (max-width: 560px) {{
      main {{ width: min(100% - 20px, 940px); margin: 20px auto; }}
      .quiz-card {{ padding: 17px; border-radius: 12px; }}
      code {{ width: 100%; margin-left: 0; }}
    }}
    @media print {{
      body {{ background: #fff; }}
      main {{ width: 100%; margin: 0; }}
      .quiz-card {{ break-inside: avoid; box-shadow: none; }}
      details {{ display: block; }}
      details > * {{ display: block; }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="page-header">
      <h1>{_escape(title)}</h1>
      <p class="summary">총 {len(quizzes)}문항 · 각 문제의 정답과 해설을 펼쳐서 확인할 수 있습니다.</p>
    </header>
    <section class="quiz-list" aria-label="생성된 퀴즈">
      {"".join(cards) if cards else '<p>생성된 퀴즈가 없습니다.</p>'}
    </section>
  </main>
</body>
</html>
"""


def visualize_quizzes(
    quizzes: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "최종 생성 퀴즈",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_quizzes(quizzes, title), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--title", default="최종 생성 퀴즈")
    args = parser.parse_args()

    output = visualize_quizzes(
        load_quizzes(args.input),
        args.output,
        title=args.title,
    )
    print(f"퀴즈 시각화 저장: {output}")


if __name__ == "__main__":
    main()
