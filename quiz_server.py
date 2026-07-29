"""
quiz_server.py

퀴즈 UI(results/quiz_ui_slide_aware.html)를 브라우저에 띄우고,
응시자가 문제를 다 풀면 이름/점수를 results/rank.csv 에 누적 저장한다.

실행:
    python quiz_server.py

실행하면 브라우저가 자동으로 http://localhost:8000 을 연다.
결과는 results/rank.csv 에 저장되며, 같은 이름으로 다시 풀면 최신 점수로 갱신된다.
"""
from __future__ import annotations

import csv
import json
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results_final"
HTML_FILE = RESULTS / "quiz_ui_slide_aware.html"
RANK_FILE = RESULTS / "rank.csv"
FIELDS = ["name", "correct", "total", "percent", "datetime"]
PORT = 8000


def read_rank() -> list[dict]:
    if not RANK_FILE.exists():
        return []
    with RANK_FILE.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("name")]


def write_rank(rows: list[dict]) -> None:
    with RANK_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def upsert(name: str, correct: int, total: int) -> list[dict]:
    """같은 이름은 최신 점수로 갱신, 없으면 추가. 정답률 내림차순 정렬 후 저장."""
    rows = read_rank()
    percent = round(correct / total * 100) if total else 0
    entry = {
        "name": name,
        "correct": str(correct),
        "total": str(total),
        "percent": str(percent),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for i, row in enumerate(rows):
        if row.get("name") == name:
            rows[i] = entry
            break
    else:
        rows.append(entry)

    rows.sort(key=lambda r: (int(r["percent"]), int(r["correct"])), reverse=True)
    write_rank(rows)
    return rows


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "text/plain; charset=utf-8") -> None:
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            if not HTML_FILE.exists():
                self._send(404, f"퀴즈 파일을 찾을 수 없습니다: {HTML_FILE}")
                return
            self._send(200, HTML_FILE.read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")
        elif self.path.startswith("/rank"):
            self._send(200, json.dumps(read_rank(), ensure_ascii=False),
                       "application/json; charset=utf-8")
        else:
            self._send(404, "not found")

    def do_POST(self) -> None:
        if self.path != "/save":
            self._send(404, "not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            name = str(payload.get("name", "")).strip() or "익명"
            correct = int(payload.get("correct", 0))
            total = int(payload.get("total", 0))
        except (ValueError, json.JSONDecodeError):
            self._send(400, "bad request")
            return
        rows = upsert(name, correct, total)
        print(f"  저장됨: {name} {correct}/{total}")
        self._send(200, json.dumps({"ok": True, "rank": rows}, ensure_ascii=False),
                   "application/json; charset=utf-8")

    def log_message(self, *args) -> None:  # 요청 로그 조용히
        pass


def main() -> None:
    # Windows 콘솔(cp949)에서 한글/특수문자 출력 시 깨지지 않도록
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    RESULTS.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{PORT}"
    print(f"퀴즈 서버 시작: {url}")
    print(f"결과 저장 위치: {RANK_FILE}")
    print("종료하려면 Ctrl+C 를 누르세요.\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    server = ThreadingHTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        server.shutdown()


if __name__ == "__main__":
    main()
