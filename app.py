"""
FloodResQ AI - small web server: serves dashboard.html and a copilot chat that is CHECKED before it answers.

Needs:  flask, gunicorn, google-genai, numpy, requests (see requirements.txt)
        and floodresq.py + results.json + dashboard.html + copilot_widget.html in the same folder as this file.

Local run:   GEMINI_API_KEY=your_key  python app.py          (the key stays on the server; never put it in any file)
Render run:  gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120

How answers stay honest: the model's reply is rejected if it contains any number that is not in results.json, or states
damage / isolation as a fact, or shows no uncertainty. A rejected reply is replaced by a report computed directly from
results.json. If the model is unavailable (no key, quota, wrong model name, network), that computed report is shown instead.

/api/chat ALWAYS answers with JSON (never an HTML error page), so the browser can show a readable message.
"""
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

import floodresq as fr

try:                                       # the app must still start (and answer from results.json) without the SDK
    from google import genai
    from google.genai import types as genai_types
except Exception:                          # pragma: no cover
    genai = None
    genai_types = None

BASE = Path(__file__).resolve().parent
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
MAX_PER_IP, WINDOW_S, MAX_GLOBAL = 12, 600, 300          # per IP per 10 min; all users per hour
MODEL_TIMEOUT_MS = 25000                                  # one model call; the copilot may make up to 2 calls

app = Flask(__name__, static_folder=None)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)           # Render sits behind a proxy: get the visitor's real IP
log = logging.getLogger("floodresq.app")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_lock = threading.Lock()
_hits = defaultdict(deque)
_global = deque()

SOURCE_NOTE = {
    "llm": "Checked: every number in this answer appears in this run's results. These are satellite-based estimates, not confirmed damage.",
    "template": "Computed directly from this run's results (no language model). Estimates, not confirmed damage.",
    "template_after_rejection": "The model's answer was withheld because it failed the number/wording check. Showing the report computed from this run's results instead.",
    "template_no_model": "The language model was unavailable. Showing the report computed from this run's results instead.",
    "safe_fallback": "The detailed report could not be built, so only the stored limitations are shown. Estimates, not confirmed damage.",
}

REQUIRED_SECTIONS = ("meta", "flood", "damage", "access")


class ResultsError(Exception):
    """results.json is missing, unreadable or has the wrong shape. The message is safe to show to the user."""


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
    """Read and sanity-check results.json. Raises ResultsError with a clear message instead of a bare crash."""
    path = BASE / "results.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ResultsError("results.json was not found next to app.py on the server.")
    except json.JSONDecodeError as e:
        raise ResultsError("results.json is not valid JSON (%s)." % e)
    if not isinstance(data, dict):
        raise ResultsError("results.json must contain a JSON object.")
    missing = [k for k in REQUIRED_SECTIONS if not isinstance(data.get(k), dict)]
    if missing:
        raise ResultsError("results.json is missing the section(s): %s." % ", ".join(missing))
    return data


def _safe_fallback_text(results):
    lim = results.get("limitations")
    text = " ".join(str(x) for x in lim) if isinstance(lim, list) and lim else \
        "Educational prototype, not an operational tool; all results need field confirmation."
    return "Sorry, a detailed answer is not available right now. Limits: " + text


def make_answer(question, lang, results, llm):
    """Return {'answer': text, 'source': ...}. Never raises: model errors fall back to the computed report,
    and if even that fails a short limitations-only text is returned. `llm(system, question)` may be None or may raise."""
    out = None
    try:
        if llm is None:
            out = fr.ask_copilot(question, results, lang)
        else:
            try:
                out = fr.ask_copilot(question, results, lang, llm=llm)
            except Exception as e:                         # quota, network, wrong model name, empty reply...
                log.warning("Gemini call failed (%s: %s); using the computed report", type(e).__name__, str(e)[:300])
                out = {"answer": fr.answer_with_templates(question, results, lang), "source": "template_no_model"}
    except Exception:
        log.exception("Could not build the computed report")
        out = {"answer": _safe_fallback_text(results), "source": "safe_fallback"}
    out["answer"] = out["answer"] + "\n\n" + SOURCE_NOTE.get(out["source"], "")
    return out


def gemini_llm():
    """Return a function call(system, question) -> text, or None when no GEMINI_API_KEY / SDK is available."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        log.warning("GEMINI_API_KEY is not set: the copilot will answer from results.json only")
        return None
    if genai is None:
        log.warning("google-genai is not installed: the copilot will answer from results.json only")
        return None
    try:
        try:
            client = genai.Client(api_key=key, http_options=genai_types.HttpOptions(timeout=MODEL_TIMEOUT_MS))
        except Exception:                                   # older SDK without http_options
            client = genai.Client(api_key=key)
    except Exception as e:
        log.warning("Could not create the Gemini client (%s: %s)", type(e).__name__, str(e)[:200])
        return None

    def call(system, question):
        cfg = genai_types.GenerateContentConfig(system_instruction=system, temperature=0.2)
        resp = client.models.generate_content(model=MODEL, contents=question, config=cfg)
        text = getattr(resp, "text", None)
        if not text or not text.strip():
            raise RuntimeError("empty model reply")
        return text.strip()
    return call


_page = {}


def add_copilot_widget(html):
    """The notebook's dashboard.html has the map but NOT the chat button. Insert copilot_widget.html once, before </body>.
    A dashboard that already contains the widget is returned unchanged."""
    if "frq-launch" in html:
        return html
    widget = (BASE / "copilot_widget.html").read_text(encoding="utf-8")
    i = html.rfind("</body>")
    if i == -1:
        i = html.rfind("</html>")
    if i == -1:
        return html + widget
    return html[:i] + widget + html[i:]


@app.get("/")
def home():
    if "html" not in _page:
        _page["html"] = add_copilot_widget((BASE / "dashboard.html").read_text(encoding="utf-8"))
    return Response(_page["html"], mimetype="text/html")


@app.get("/healthz")
def healthz():
    return jsonify(status="ok")


@app.get("/api/results")
def get_results():
    return jsonify(load_results())


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send JSON like {\"question\": \"...\", \"language\": \"en\"}."), 400
    question = str(data.get("question", "")).strip()
    lang = "ne" if data.get("language") == "ne" else "en"
    if not question:
        return jsonify(error="Please enter a question."), 400
    if len(question) > 700:
        return jsonify(error="Question is too long."), 400
    if not rate_ok(request.remote_addr or "unknown"):
        return jsonify(error="Too many questions. Please wait a few minutes and try again."), 429
    results = load_results()                                # ResultsError is turned into JSON below
    out = make_answer(question, lang, results, gemini_llm())
    return jsonify(answer=out["answer"], source=out["source"])


@app.errorhandler(ResultsError)
def handle_results_error(e):
    log.error("results.json problem: %s", e)
    return jsonify(error="Server data problem: %s" % e), 500


@app.errorhandler(Exception)
def handle_any_error(e):
    """Turn every crash into JSON for /api/* (HTML only for normal pages), and log the full traceback for Render."""
    if isinstance(e, HTTPException):
        if request.path.startswith("/api/"):
            return jsonify(error="%s %s" % (e.code, e.name)), e.code
        return e
    log.exception("Unhandled error on %s %s", request.method, request.path)
    if request.path.startswith("/api/"):
        return jsonify(error="Internal server error. The site owner can find details in the server logs."), 500
    return Response("Internal server error", status=500, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
