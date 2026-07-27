"""
src/evaluator.py

grounded_score: 생성된 답변(evidence)이 실제로 검색된 원문 청크에 포함되는지
                문자열 유사도로 체크 (LLM 없이 순수 텍스트 연산이라 이 샌드박스에서도
                바로 검증 가능)
bert_score_eval: 생성된 답변 vs 정답(ground truth) 사이의 BERTScore F1
                 (bert-score 라이브러리가 내부적으로 사전학습 모델(기본값
                 roberta-large, 다국어는 bert-base-multilingual-cased 지정 필요)을
                 huggingface에서 내려받아야 해서, 이 샌드박스에서는 실행 불가 ->
                 코드는 정상 동작하는 형태로 작성해두고, 실행은 인터넷 되는
                 환경에서 진행)
"""
import difflib


def grounded_score(evidence: str, source_chunks: list[str], threshold: float = 0.6) -> float:
    """
    evidence(생성모델이 근거라고 제시한 문장)가 실제 검색된 청크들 중 어딘가에
    실제로 포함/유사한지를 SequenceMatcher 유사도로 채점.
    threshold 이상이면 "근거 있음"으로 보고 1.0, 아니면 최고 유사도 값을 그대로 반환
    (0~1 사이 연속값이라 문항별 평균을 내면 코퍼스 전체의 '근거성 점수'가 됨).
    """
    best = 0.0
    for chunk in source_chunks:
        ratio = difflib.SequenceMatcher(None, evidence, chunk).ratio()
        best = max(best, ratio)
    return 1.0 if best >= threshold else best


def bert_score_eval(candidates: list[str], references: list[str], lang: str = "ko"):
    """
    [샌드박스에서 실행 불가 - huggingface 모델 다운로드 필요]
    실제 실행 환경에서는 아래처럼 그대로 쓰면 됨:

        from bert_score import score
        P, R, F1 = score(candidates, references, lang=lang,
                          model_type="bert-base-multilingual-cased")
        return {"precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item()}

    한국어라 lang="ko" 지정 시 bert-score가 자동으로
    다국어 모델(bert-base-multilingual-cased)을 골라줌.
    """
    raise RuntimeError(
        "bert-score는 huggingface 모델 다운로드가 필요해 이 샌드박스에서 실행 불가. "
        "인터넷이 되는 환경(다른 AI/로컬 PC)에서 위 주석의 코드로 실행할 것."
    )


if __name__ == "__main__":
    # grounded_score는 모델 없이 바로 검증 가능
    source_chunks = [
        "2015.07 부산시 기장군 자동차 부품공장. 피자재 금형 사이에 상체 넣어 금형 점검. "
        "동료 작업자가 프레스를 조작하여 피재자의 상체가 금형에 협착",
        "안전장치 부착. 작업 시작 전 안전점검. 고소작업 안전장비 착용",
    ]

    # 케이스 1: 실제 원문과 거의 같은 근거 -> 높은 점수 기대
    evidence_good = "2015년 7월 부산시 기장군 자동차 부품공장에서 동료 작업자가 프레스를 조작해 협착 사고 발생"
    # 케이스 2: 원문과 무관한 지어낸 근거 -> 낮은 점수 기대
    evidence_bad = "2020년 서울 강남구 공장에서 화재 사고가 발생했다"

    print("근거 있는 답변 grounded_score:", round(grounded_score(evidence_good, source_chunks), 3))
    print("지어낸 답변 grounded_score   :", round(grounded_score(evidence_bad, source_chunks), 3))
