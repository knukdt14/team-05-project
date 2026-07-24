"""
src/documents.py
pptx 파일에서 슬라이드별로 텍스트를 추출한다.
- python-pptx로 각 슬라이드의 모든 텍스트 도형(title, body, table, textbox)을 순회
- 표(table)는 행 단위로 " | "로 이어붙여서 표 구조가 최대한 유지되도록 함
"""
from pathlib import Path
from pptx import Presentation


def extract_slide_texts(pptx_path: str) -> list[dict]:
    """슬라이드별 {slide_no, text} 딕셔너리 리스트를 반환."""
    prs = Presentation(pptx_path)
    results = []

    for i, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs)
                    if line.strip():
                        parts.append(line.strip())
            elif shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))

        text = "\n".join(parts)
        results.append({"slide_no": i, "text": text})

    return results


if __name__ == "__main__":
    docs = extract_slide_texts("../../deck.pptx")
    print(f"총 {len(docs)}개 슬라이드 추출")
    print(docs[8])  # 슬라이드 9번 미리보기
