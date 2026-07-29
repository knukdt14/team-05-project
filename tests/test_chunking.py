from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunking import chunk_documents  # noqa: E402


class LayoutAwareChunkingTests(unittest.TestCase):
    def test_repeats_title_and_table_header(self) -> None:
        documents = [
            {
                "slide_no": 3,
                "title": "프레스 가공",
                "text": "프레스 가공\n공정 | 설명\n피어싱 | 구멍을 냄",
                "body_groups": [
                    {
                        "type": "table",
                        "text": "공정 | 설명\n피어싱 | 구멍을 냄\n샤링 | 절단함",
                        "lines": [
                            "공정 | 설명",
                            "피어싱 | 구멍을 냄",
                            "샤링 | 절단함",
                        ],
                    }
                ],
            }
        ]

        chunks = chunk_documents(
            documents,
            "layout_aware",
            chunk_size=45,
            overlap=0,
            document_id="test",
        )

        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertIn("[제목] 프레스 가공", chunk["text"])
            self.assertIn("[표 헤더] 공정 | 설명", chunk["text"])
            self.assertEqual(chunk["chunk_method"], "layout_aware")

    def test_rejects_invalid_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            chunk_documents([], "sentence_pack", chunk_size=0)


if __name__ == "__main__":
    unittest.main()
