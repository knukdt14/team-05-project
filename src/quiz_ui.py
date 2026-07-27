"""
src/quiz_ui.py

results/generated_quizzes.json 을 읽어, 실제로 풀어보는 인터랙티브 퀴즈
웹페이지(results/quiz_ui.html)를 생성한다. (서버 불필요, 브라우저로 바로 열기)

기존 visualize_quizzes.py = 정답이 다 보이는 '검토용'.
이 파일 = 보기를 클릭해 풀고 채점되는 '응시용' UI.

실행:  cd src && python -m quiz_ui
"""
from __future__ import annotations

import json
from pathlib import Path

from settings import PROJECT_ROOT

DEFAULT_INPUT = PROJECT_ROOT / "results" / "generated_quizzes.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "quiz_ui.html"


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>신입사원 교육 퀴즈</title>
<style>
  :root { --bg:#f4f6fb; --card:#fff; --ink:#1f2430; --muted:#6b7280;
          --brand:#3b6ef5; --ok:#16a34a; --no:#dc2626; --line:#e5e7eb; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:'Malgun Gothic','Apple SD Gothic Neo',system-ui,sans-serif; }
  .wrap { max-width:720px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:22px; margin:8px 0 4px; }
  .sub { color:var(--muted); font-size:14px; margin-bottom:20px; }
  .bar { height:8px; background:var(--line); border-radius:99px; overflow:hidden; margin-bottom:20px; }
  .bar > i { display:block; height:100%; background:var(--brand); width:0; transition:width .3s; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:16px;
          padding:24px; box-shadow:0 2px 10px rgba(0,0,0,.04); }
  .meta { color:var(--muted); font-size:13px; margin-bottom:8px; }
  .q { font-size:18px; font-weight:600; line-height:1.5; margin-bottom:20px; }
  .choice { display:block; width:100%; text-align:left; background:#fff; color:var(--ink);
            border:1.5px solid var(--line); border-radius:12px; padding:14px 16px;
            margin-bottom:10px; font-size:15px; cursor:pointer; transition:.15s; }
  .choice:hover:not(:disabled) { border-color:var(--brand); background:#f5f8ff; }
  .choice .n { display:inline-block; width:24px; height:24px; line-height:24px; text-align:center;
               border-radius:50%; background:var(--line); margin-right:10px; font-size:13px; }
  .choice.ok { border-color:var(--ok); background:#eafaf0; }
  .choice.no { border-color:var(--no); background:#fdecec; }
  .choice.ok .n { background:var(--ok); color:#fff; }
  .choice.no .n { background:var(--no); color:#fff; }
  .reveal { margin-top:16px; padding:16px; border-radius:12px; background:#f8fafc;
            border:1px solid var(--line); font-size:14px; line-height:1.6; display:none; }
  .reveal.show { display:block; }
  .reveal b { color:var(--brand); }
  .src { color:var(--muted); font-size:12px; margin-top:8px; }
  .row { display:flex; justify-content:space-between; align-items:center; margin-top:20px; }
  .btn { background:var(--brand); color:#fff; border:none; border-radius:10px;
         padding:12px 22px; font-size:15px; cursor:pointer; }
  .btn:disabled { opacity:.4; cursor:not-allowed; }
  .score { text-align:center; padding:40px 20px; }
  .score .big { font-size:44px; font-weight:800; color:var(--brand); margin:8px 0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1420; --card:#171d2b; --ink:#e8eaf0; --muted:#9aa3b2; --line:#2a3346; }
    .choice { background:#171d2b; } .choice:hover:not(:disabled){ background:#1c2740; }
    .reveal { background:#131a28; } .choice.ok{background:#12261b;} .choice.no{background:#2a1618;}
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>신입사원 교육 퀴즈</h1>
  <div class="sub" id="sub"></div>
  <div class="bar"><i id="prog"></i></div>
  <div id="app"></div>
</div>
<script>
const QUIZZES = __QUIZ_DATA__;
let idx = 0, score = 0, answered = false;

function render() {
  const app = document.getElementById('app');
  document.getElementById('sub').textContent =
    `문제 ${idx+1} / ${QUIZZES.length}`;
  document.getElementById('prog').style.width =
    `${(idx)/QUIZZES.length*100}%`;

  if (idx >= QUIZZES.length) { renderScore(); return; }
  const q = QUIZZES[idx];
  const correct = (q.answer|0) - 1;
  answered = false;

  let html = `<div class="card"><div class="meta">${q.type||''}</div>`;
  html += `<div class="q">${esc(q.question)}</div>`;
  (q.choices||[]).forEach((c, i) => {
    html += `<button class="choice" data-i="${i}">
      <span class="n">${i+1}</span>${esc(c)}</button>`;
  });
  html += `<div class="reveal" id="reveal">
      <div><b>해설</b> ${esc(q.explanation||'')}</div>
      <div style="margin-top:8px"><b>근거</b> ${esc(q.evidence||'')}</div>
      <div class="src">${srcText(q.sources)}</div></div>`;
  html += `<div class="row">
      <span id="fb" style="font-weight:600"></span>
      <button class="btn" id="next" disabled>다음 ›</button></div></div>`;
  app.innerHTML = html;

  app.querySelectorAll('.choice').forEach(btn => {
    btn.onclick = () => pick(parseInt(btn.dataset.i), correct);
  });
  document.getElementById('next').onclick = () => { idx++; render(); };
}

function pick(i, correct) {
  if (answered) return;
  answered = true;
  const btns = document.querySelectorAll('.choice');
  btns.forEach(b => b.disabled = true);
  btns[correct].classList.add('ok');
  if (i === correct) { score++; document.getElementById('fb').textContent = '정답! ✅'; }
  else { btns[i].classList.add('no');
         document.getElementById('fb').textContent = '오답 ❌'; }
  document.getElementById('reveal').classList.add('show');
  document.getElementById('next').disabled = false;
}

function renderScore() {
  document.getElementById('prog').style.width = '100%';
  const pct = Math.round(score/QUIZZES.length*100);
  document.getElementById('app').innerHTML = `<div class="card score">
    <div>완료!</div><div class="big">${score} / ${QUIZZES.length}</div>
    <div style="color:var(--muted)">정답률 ${pct}%</div>
    <div class="row" style="justify-content:center;margin-top:24px">
      <button class="btn" onclick="idx=0;score=0;render()">다시 풀기</button>
    </div></div>`;
}

function srcText(sources) {
  if (!Array.isArray(sources)) return '';
  return '출처 · ' + sources.map(s =>
    `슬라이드 ${s.slide}`).join(' / ');
}
function esc(s){ return String(s).replace(/[&<>]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

render();
</script>
</body>
</html>
"""


def build_ui(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT) -> Path:
    quizzes = json.loads(Path(input_path).read_text(encoding="utf-8"))
    data = json.dumps(quizzes, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__QUIZ_DATA__", data)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    path = build_ui()
    print(f"퀴즈 UI 생성: {path}")
    print("브라우저로 이 파일을 열면 퀴즈를 풀 수 있습니다.")
