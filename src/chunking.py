"""
src/chunking.py
글자수 150 기준 recursive 청킹.
- 우선 문단(\n\n 또는 \n) 단위로 쪼개려고 시도
- 그래도 150자를 넘으면 문장(다./함./음. 등) 단위로 재분할
- 그래도 넘으면 글자수로 강제 분할
- overlap: 앞 청크 끝 30자를 다음 청크 시작에 겹쳐서 문맥 단절 완화
"""
import re

CHUNK_SIZE = 150
OVERLAP = 30


def _split_by_sentence(text: str) -> list[str]:
    # 한국어 문장 종결 패턴 기준 분할 (마침표/물음표/느낌표 뒤)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s.strip()]


def recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    paragraphs = [p for p in text.split("\n") if p.strip()]

    # 1) 문단들을 chunk_size 이하로 최대한 묶어봄
    chunks = []
    buf = ""
    for para in paragraphs:
        candidate = (buf + "\n" + para).strip() if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # 문단 자체가 chunk_size보다 길면 문장 단위로 재귀 분할
            if len(para) > chunk_size:
                chunks.extend(_split_by_sentence_to_size(para, chunk_size))
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)

    # 2) overlap 적용
    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0 and overlap > 0:
            prev_tail = chunks[i - 1][-overlap:]
            c = prev_tail + " " + c
        overlapped.append(c)

    return overlapped


def _split_by_sentence_to_size(text: str, chunk_size: int) -> list[str]:
    sentences = _split_by_sentence(text)
    chunks, buf = [], ""
    for s in sentences:
        candidate = (buf + " " + s).strip() if buf else s
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # 문장 하나가 chunk_size보다 길면 글자수로 강제 분할
            if len(s) > chunk_size:
                for j in range(0, len(s), chunk_size):
                    chunks.append(s[j:j + chunk_size])
                buf = ""
            else:
                buf = s
    if buf:
        chunks.append(buf)
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    """[{slide_no, text}] -> [{chunk_id, slide_no, text}]"""
    all_chunks = []
    cid = 0
    for doc in docs:
        for piece in recursive_chunk(doc["text"]):
            all_chunks.append({
                "chunk_id": f"c{cid:04d}",
                "slide_no": doc["slide_no"],
                "text": piece,
            })
            cid += 1
    return all_chunks


if __name__ == "__main__":
    from documents import extract_slide_texts
    docs = extract_slide_texts("../../deck.pptx")
    chunks = chunk_documents(docs)
    print(f"총 {len(docs)}개 슬라이드 -> {len(chunks)}개 청크 (chunk_size={CHUNK_SIZE})")
    lens = [len(c["text"]) for c in chunks]
    print(f"청크 길이 평균 {sum(lens)/len(lens):.1f}자, 최대 {max(lens)}자, 최소 {min(lens)}자")
    for c in chunks[15:18]:
        print(c)
