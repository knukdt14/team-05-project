"""
ai_judge.py

사람 채점자와 똑같은 2개 기준(명확성, 교육적가치)으로, 로컬 Qwen 모델이
퀴즈를 0/1로 자동 채점하게 하는 스크립트. quiz_generator.py의 기존
패턴(load_llm, extract_json)을 그대로 재사용해서 별도 의존성을 안 늘림.

결과를 results/ai_judgments.json에 저장 -> kappa_eval.py가 이 파일과
사람 채점 엑셀 3개를 같이 읽어서 일치도를 계산함.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quiz_generator import extract_json, load_llm, _invoke

JUDGE_SYSTEM_PROMPT = """\
너는 신입사원 교육용 퀴즈를 검수하는 평가자다.
주어진 퀴즈 하나를 아래 2개 기준으로 각각 0 또는 1로 채점하라.

1) 명확성: 질문이 무엇을 묻는지 분명하고, 보기 중 정답이 명확히 구분되면 1,
   질문이 모호하거나 문장이 어색해서 뭘 묻는지 헷갈리면 0.
2) 교육적가치: 신입사원이 실무에서 실제로 알아야 할 핵심 내용을 다루면 1,
   지엽적이거나 실무와 관련 없는 사소한 디테일이면 0.

설명 없이 다음 형식의 JSON 하나만 출력하라:
{"명확성": 0 또는 1, "교육적가치": 0 또는 1}
"""


def _quiz_text(quiz: dict) -> str:
    choices = quiz.get("choices") or []
    choices_text = "\n".join(f"{i}. {c}" for i, c in enumerate(choices, start=1)) or "(단답형)"
    answer = quiz.get("answer", "")
    answer_text = (
        f"{answer}. {choices[answer - 1]}"
        if isinstance(answer, int) and 1 <= answer <= len(choices)
        else str(answer)
    )
    return (
        f"질문: {quiz.get('question', '')}\n"
        f"보기:\n{choices_text}\n"
        f"정답: {answer_text}\n"
        f"해설: {quiz.get('explanation', '')}\n"
        f"근거: {quiz.get('evidence', '')}"
    )


def judge_quiz(generator, quiz: dict) -> dict[str, int]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _quiz_text(quiz)},
    ]
    raw = _invoke(generator, messages, max_new_tokens=100)
    result = extract_json(raw)
    return {
        "명확성": int(result["명확성"]),
        "교육적가치": int(result["교육적가치"]),
    }


def judge_all(generator, quizzes: list[dict]) -> list[dict[str, Any]]:
    results = []
    for i, quiz in enumerate(quizzes, start=1):
        entry = {"quiz_id": quiz.get("quiz_id", f"quiz-{i}")}
        try:
            entry.update(judge_quiz(generator, quiz))
        except Exception as error:  # noqa: BLE001 - 하나 실패해도 계속 진행
            entry["명확성"] = None
            entry["교육적가치"] = None
            entry["error"] = str(error)
            print(f"[{i}/{len(quizzes)}] {entry['quiz_id']}: 실패 - {error!r}")
        else:
            print(f"[{i}/{len(quizzes)}] {entry['quiz_id']}: "
                  f"명확성={entry['명확성']} 교육적가치={entry['교육적가치']}")
        results.append(entry)
    return results


if __name__ == "__main__":
    import sys

    quizzes_path = sys.argv[1] if len(sys.argv) > 1 else r"..\results\generated_quizzes.json"
    output_path = "../results/ai_judgments.json"

    with open(quizzes_path, encoding="utf-8") as f:
        quizzes = json.load(f)
    print(f"퀴즈 {len(quizzes)}개 로딩됨")

    print("AI 채점 모델 로딩 중 (Qwen)...")
    generator = load_llm()

    judgments = judge_all(generator, quizzes)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judgments, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {output_path}")
