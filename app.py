"""
FloodResQ AI - small web server: serves dashboard.html and a copilot chat that is CHECKED before it answers.

Needs:  pip install flask google-genai   and   floodresq.py (version 0.2.3) + results.json + dashboard.html in this folder.
Run:    GEMINI_API_KEY=...  python app.py        (the key stays on the server; never put it in any file)

How answers stay honest: the model's reply is rejected if it contains any number that is not in results.json, or states
damage / isolation as a fact, or shows no uncertainty. A rejected reply is replaced by a report computed directly from
results.json. If the model is unavailable, that computed report is shown instead.
"""
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, Response, jsonify, request
from google import genai

import floodresq as fr

BASE = Path(__file__).resolve().parent
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")     # check the model name against your API's model list
MAX_PER_IP, WINDOW_S, MAX_GLOBAL = 12, 600, 300                       # per IP per 10 min; all users per hour
app = Flask(__name__, static_folder=None)
_lock = threading.Lock()
_hits = defaultdict(deque)
_global = deque()

SOURCE_NOTE = {
    "llm": "Checked: every number in this answer appears in this run's results. These are satellite-based estimates, not confirmed damage.",
    "template": "Computed directly from this run's results (no language model). Estimates, not confirmed damage.",
    "template_after_rejection": "The model's answer was withheld because it failed the number/wording check. Showing the report computed from this run's results instead.",
    "template_no_model": "The language model was unavailable. Showing the report computed from this run's results instead.",
}


def rate_ok(ip, now=None):
    """True if this caller may ask now (per-IP and global limits, in memory)."""
    now = time.time() if now is None else now
    with _lock:
        q = _hits[ip]
        while q and now - q[0] > WINDOW_S:
            q.popleft()
        while _global and now - _global[0] > 3600:
            _global.popleft()
        if len(q) >= MAX_PER_IP or len(_global) >= MAX_GLOBAL:
            return False
        q.append(now)
        _global.append(now)
        return True


def load_results():
    with open(BASE / "results.json", encoding="utf-8") as f:
        return json.load(f)


def make_answer(question, lang, results, llm):
    """Return {'answer': text, 'source': ...}. `llm(system, question)` may be None or may raise."""
    if llm is None:
        out = fr.ask_copilot(question, results, lang)
    else:
        try:
            out = fr.ask_copilot(question, results, lang, llm=llm)
        except Exception:                                  # quota, network, model name...
            out = {"answer": fr.answer_with_templates(question, results, lang), "source": "template_no_model"}
    out["answer"] = out["answer"] + "\n\n" + SOURCE_NOTE.get(out["source"], "")
    return out


def gemini_llm():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    client = genai.Client(api_key=key)

    def call(system, question):
        resp = client.models.generate_content(model=MODEL, contents=system + "\n\nQUESTION: " + question)
        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("empty model reply")
        return text
    return call


_page = {}


@app.get("/")
def home():
    if "html" not in _page:
        html = (BASE / "dashboard.html").read_text(encoding="utf-8")
        html = html.replace("Answers based on this run's results",
                            "Answers are checked against this run's results; estimates, not confirmed damage")
        _page["html"] = html
    return Response(_page["html"], mimetype="text/html")


@app.get("/api/results")
def get_results():
    return jsonify(load_results())


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    lang = "ne" if data.get("language") == "ne" else "en"
    if not question:
        return jsonify(error="Please enter a question."), 400
    if len(question) > 700:
        return jsonify(error="Question is too long."), 400
    if not rate_ok(request.remote_addr or "unknown"):
        return jsonify(error="Too many questions. Please wait a few minutes and try again."), 429
    out = make_answer(question, lang, load_results(), gemini_llm())
    return jsonify(answer=out["answer"], source=out["source"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
