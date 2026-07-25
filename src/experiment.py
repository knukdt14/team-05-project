"""
src/experiment.py  [C 담당: 모델 비교 실험 - 풀세트]

모델(Qwen/Solar...) × RAG(on/off) 를 여러 주제로 자동 실행하고,
평가지표를 모아 '모델별 점수표'로 출력/저장한다. -> 가장 좋은 LLM 확정용.

측정 지표:
  - json_ok      : 생성 결과가 형식(JSON dict) 안 깨지고 나온 비율 (parse_ok 평균)
  - grounded     : evidence가 실제 정답 슬라이드 원문에 있는지 (근거성, 0~1)
  - bertscore    : 생성한 퀴즈 내용이 원문과 의미적으로 가까운지 (F1, 0~1)
  - sec/quiz     : 문항당 평균 생성 시간(초)

실행:  (.venv 켜고) cd src && python experiment.py
결과:  콘솔 표 + results/comparison.csv + results/quizzes_*.json 저장
"""
import time
import json
import gc
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from documents import extract_slide_texts
from chunking import chunk_documents
from embeddings import ProductionEmbedder
from index import build_index
from retriever import retrieve
from generator_lc import build_generator, generate_quiz
from evaluator import grounded_score

PPTX_PATH = "../data/deck.pptx"
TOP_K = 3
RESULTS_DIR = Path("../results")

# ---- 비교할 모델들 (이 목록만 바꾸면 됨) ----
# [2단계] 1단계에서 Qwen-3B 확정 -> 이 모델로 RAG 효과(켬/끔) 검증
CONFIGS = [
    {"name": "qwen-3b",    "provider": "hf",      "model": "Qwen/Qwen2.5-3B-Instruct"},
    # --- 1단계(모델선정)에서 쓴 목록. 다시 비교할 때 주석 풀기 ---
    # {"name": "solar-mini", "provider": "upstage", "model": "solar-mini"},
    # {"name": "solar-pro",  "provider": "upstage", "model": "solar-pro"},
    # {"name": "qwen-1.5b",  "provider": "hf",      "model": "Qwen/Qwen2.5-1.5B-Instruct"},
]
# 2단계 핵심: RAG 켬 vs 끔 비교
RAG_MODES = [True, False]

# ---- 평가셋 (주제 + 정답 슬라이드). 늘릴수록 통계적으로 신뢰도 ↑ ----
# n=5 -> 16으로 확장 (안전관리 35~40 + 생산관리 41~45 커버)
EVAL_SET = [
    # 안전관리 섹션
    {"topic": "크레인 작업 시 위험발생요인과 대책",        "relevant_slides": {38}},
    {"topic": "크레인 협착 사고 방지 대책",              "relevant_slides": {38}},
    {"topic": "프레스 작업 중 재해사례",                 "relevant_slides": {39}},
    {"topic": "프레스 협착 방지 방호장치",               "relevant_slides": {39}},
    {"topic": "하인리히 법칙의 재해 비율(1:29:300)",     "relevant_slides": {36}},
    {"topic": "하인리히 법칙에서 아차사고(Near Miss)의 의미", "relevant_slides": {36}},
    {"topic": "지게차 작업 시 위험요소와 대책",           "relevant_slides": {37}},
    {"topic": "지게차 급가속·급선회로 인한 사고",         "relevant_slides": {37}},
    {"topic": "물질안전보건자료(MSDS)에 명시되는 내용",    "relevant_slides": {40}},
    {"topic": "화학물질 취급 시 사고 사례",              "relevant_slides": {40}},
    {"topic": "재해발생 이론에서 불안전한 행동과 상태",     "relevant_slides": {35}},
    # 생산관리 섹션
    {"topic": "PDCA 사이클의 단계와 의미",               "relevant_slides": {41}},
    {"topic": "생산관리의 7대 낭비",                     "relevant_slides": {42}},
    {"topic": "3정(정품·정량·정위치)의 개념",            "relevant_slides": {43}},
    {"topic": "5S(정리·정돈·청소·청결·생활화)",          "relevant_slides": {44}},
    {"topic": "생산관리 프로세스에서 QCD의 의미",         "relevant_slides": {45}},
]


def _bertscore(candidates: list[str], references: list[str]) -> float:
    """생성 퀴즈 내용(candidate) vs 정답 슬라이드 원문(reference)의 BERTScore F1 평균.
    (평가 담당 evaluator.bert_score_eval와 동일 로직. bert-score 라이브러리 사용)"""
    if not candidates:
        return 0.0
    try:
        from bert_score import score
        P, R, F1 = score(candidates, references, lang="ko",
                         model_type="bert-base-multilingual-cased", verbose=False)
        return round(F1.mean().item(), 3)
    except Exception as e:
        print(f"  [BERTScore 건너뜀: {e}]")
        return float("nan")


def run_config(cfg: dict, use_rag: bool, collection, embedder, slide_texts: dict) -> dict:
    """한 모델 x RAG모드로 EVAL_SET 전체를 돌리고 지표를 집계한다."""
    label = f"{cfg['name']}{'+RAG' if use_rag else '(baseline)'}"
    print(f"\n▶ {label} 실행 중...")

    try:
        chain = build_generator(cfg["provider"], cfg["model"],
                                temperature=0.0, use_rag=use_rag)
    except Exception as e:
        print(f"  ✖ 모델 로드 실패: {e}")
        return {"label": label, "error": str(e)}

    quizzes, ok_flags, grounded_list = [], [], []
    cand_texts, ref_texts = [], []
    t_total = 0.0

    for i, case in enumerate(EVAL_SET, start=1):
        topic = case["topic"]
        retrieved = retrieve(collection, embedder, topic, k=TOP_K) if use_rag else None
        # 정답 슬라이드 원문(채점 기준 reference)
        ref = " ".join(t for s in case["relevant_slides"] for t in slide_texts.get(s, []))

        t0 = time.perf_counter()
        quiz = generate_quiz(chain, topic, retrieved, quiz_id=f"quiz-{i:03d}")
        t_total += time.perf_counter() - t0

        quizzes.append(quiz)
        ok = quiz.get("parse_ok", False)
        ok_flags.append(1 if ok else 0)

        if ok:
            # 근거성: evidence가 정답 슬라이드 원문에 있나
            ev = quiz.get("evidence", "")
            grounded_list.append(grounded_score(ev, slide_texts_list(case, slide_texts)))
            # BERTScore용: 생성한 문제+해설 vs 정답 원문
            cand_texts.append(f"{quiz.get('question','')} {quiz.get('explanation','')}")
            ref_texts.append(ref)

    n = len(EVAL_SET)
    metrics = {
        "label": label,
        "json_ok": round(sum(ok_flags) / n, 3),
        "grounded": round(sum(grounded_list) / len(grounded_list), 3) if grounded_list else 0.0,
        "bertscore": _bertscore(cand_texts, ref_texts),
        "sec_per_quiz": round(t_total / n, 2),
        "n": n,
    }

    # 생성물 저장 (수동 평가/검수용)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"quizzes_{cfg['name']}{'_rag' if use_rag else '_base'}.json"
    out.write_text(json.dumps(quizzes, ensure_ascii=False, indent=2), encoding="utf-8")

    # HF 로컬 모델은 VRAM 정리 (다음 모델 로드 전)
    del chain
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"  ✔ json_ok={metrics['json_ok']} grounded={metrics['grounded']} "
          f"bertscore={metrics['bertscore']} {metrics['sec_per_quiz']}s/개")
    return metrics


def slide_texts_list(case: dict, slide_texts: dict) -> list[str]:
    """정답 슬라이드들의 청크 텍스트 리스트 (grounded_score용)."""
    out = []
    for s in case["relevant_slides"]:
        out.extend(slide_texts.get(s, []))
    return out


def main():
    print("=" * 60)
    print("모델 비교 실험 (모델 × RAG × 주제)")
    print("=" * 60)

    # 1) 인덱스 1회 구축 (모델과 무관하므로 공유)
    print("인덱스 구축 중...")
    docs = extract_slide_texts(PPTX_PATH)
    chunks = chunk_documents(docs)
    embedder = ProductionEmbedder("intfloat/multilingual-e5-small")
    collection = build_index(chunks, embedder)

    # 슬라이드별 원문 텍스트 맵 (채점 기준)
    slide_texts: dict[int, list[str]] = {}
    for c in chunks:
        slide_texts.setdefault(c["slide_no"], []).append(c["text"])
    print(f"완료: 슬라이드 {len(docs)}개 -> 청크 {len(chunks)}개")

    # 2) 모든 조합 실행
    rows = []
    for use_rag in RAG_MODES:
        for cfg in CONFIGS:
            rows.append(run_config(cfg, use_rag, collection, embedder, slide_texts))

    # 3) 결과표 출력
    print("\n" + "=" * 72)
    print(f"{'모델':22s} {'JSON성공':>8s} {'근거성':>7s} {'BERTScore':>10s} {'초/개':>7s}")
    print("-" * 72)
    for r in rows:
        if "error" in r:
            print(f"{r['label']:22s} {'ERROR: ' + r['error'][:40]}")
        else:
            print(f"{r['label']:22s} {r['json_ok']:>8.3f} {r['grounded']:>7.3f} "
                  f"{r['bertscore']:>10} {r['sec_per_quiz']:>7.2f}")
    print("=" * 72)

    # 4) CSV 저장
    RESULTS_DIR.mkdir(exist_ok=True)
    csv = RESULTS_DIR / "comparison.csv"
    header = "label,json_ok,grounded,bertscore,sec_per_quiz,n\n"
    lines = [header]
    for r in rows:
        if "error" not in r:
            lines.append(f"{r['label']},{r['json_ok']},{r['grounded']},"
                         f"{r['bertscore']},{r['sec_per_quiz']},{r['n']}\n")
    csv.write_text("".join(lines), encoding="utf-8")
    print(f"\n저장됨: {csv}  (생성물은 results/quizzes_*.json)")


if __name__ == "__main__":
    main()
