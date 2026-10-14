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
import html as html_lib
import json
import logging
import os
import re
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
    problems = fr.validate_results(data)
    if problems and not fr.is_legacy_results(data):
        # current layout but incomplete/invalid: refuse instead of answering from a half-broken file
        raise ResultsError("results.json does not match the expected schema: %s" % "; ".join(problems[:4]))
    if problems:
        log.warning("results.json uses an older layout (%s); answers will say so", problems[0])
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


_llm_cache = {}


def gemini_llm():
    """Cached wrapper: build the Gemini client once per process (and again only if the key changes)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key not in _llm_cache:
        _llm_cache.clear()
        _llm_cache[key] = _build_gemini_llm()
    return _llm_cache[key]


def _build_gemini_llm():
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

NOTES_CSS = ("#frq-notes{position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9998;width:min(520px,calc(100vw - 460px));"
             "min-width:260px;background:#fff;border:2px solid __COLOR__;border-radius:8px;font:12px/1.4 Arial,sans-serif;color:#172033;"
             "box-shadow:0 2px 10px #0003}#frq-notes summary{cursor:pointer;padding:7px 10px;font-weight:700}"
             "#frq-notes div{padding:0 10px 8px}@media(max-width:900px){#frq-notes{left:8px;transform:none;width:calc(100vw - 16px);top:auto;bottom:60px}}")
VERDICT_COLOR = {"water_detected": "#1e6fd9", "no_water_detected": "#2e7d32", "inconclusive": "#b26a00"}


def _reasons_text(results):
    ev = results.get("evidence") or {}
    return "; ".join(fr._reason_text(r, "en") for r in ev.get("reasons", []) if r.get("code") != "no_control")


def build_notes_panel(results, mismatches):
    """A small, data-driven box at the top of the map: verdict and how to read the imagery. Everything comes from results.json."""
    ev = results.get("evidence") or {}
    verdict = fr.verdict_text(results)
    color = VERDICT_COLOR.get(ev.get("verdict"), "#555")
    fl = results.get("flood") or {}
    reasons = _reasons_text(results)
    warn = ""
    if mismatches:
        warn = ("<div style='background:#fdecea;border:1px solid #c62828;padding:6px;margin:4px 10px'><b>Warning:</b> the map and "
                "results.json may come from different runs (%s). Re-run the notebook and upload both files together.</div>"
                % html_lib.escape("; ".join(mismatches)))
    cand = ""
    if ev.get("verdict") == "inconclusive" and fl.get("water_km2"):
        cand = " Radar candidate: %s sq km (unconfirmed)." % fr._f(fl["water_km2"])
    return ("<style>" + NOTES_CSS.replace("__COLOR__", color) + "</style><details id='frq-notes' open><summary>Verdict: " + html_lib.escape(verdict) +
            "</summary>" + warn + "<div>" + html_lib.escape(reasons[:1].upper() + reasons[1:] + "." if reasons else "") +
            html_lib.escape(cand) + "<br><b>How to read the map:</b> radar layers are grey scale (dark = low backscatter, which can be "
            "water, but also smooth ground or radar shadow; transparent = excluded pixels). Sentinel-2 layers are true colour "
            "(white = cloud, transparent = no data). Coloured polygons are algorithm candidates, not confirmed flooding or damage. "
            "No comparison with an independent reference map has been run.</div></details>")


def sync_dashboard(html, results):
    """Make the stored map agree with results.json: replace the verdict line with the one computed from results.json,
    and show a visible warning when the map and the results look like different runs."""
    mismatches = fr.dashboard_matches_results(html, results)
    verdict = fr.verdict_text(results)
    reasons = _reasons_text(results)
    line = "<b>Verdict:</b> %s%s<br>" % (html_lib.escape(verdict), (" (" + html_lib.escape(reasons) + ")") if reasons else "")
    html, n = re.subn(r"<b>Verdict:</b>.*?<br>", lambda m: line, html, count=1, flags=re.S)
    return html, build_notes_panel(results, mismatches), mismatches


def add_copilot_widget(html, notes=""):
    """The notebook's dashboard.html has the map but NOT the chat button. Insert the notes panel and
    copilot_widget.html once, before </body>. A page that already contains the widget is returned unchanged."""
    if "frq-launch" in html:
        return html
    widget = (BASE / "copilot_widget.html").read_text(encoding="utf-8")
    extra = notes + widget
    i = html.rfind("</body>")
    if i == -1:
        i = html.rfind("</html>")
    if i == -1:
        return html + extra
    return html[:i] + extra + html[i:]


def _plain_page(title, text, status):
    body = ("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>%s</title><body style='font:16px Arial;max-width:640px;margin:40px auto;padding:0 16px'><h1>%s</h1><p>%s</p>"
            "<p><a href='/healthz'>Health check</a></p></body>" % (html_lib.escape(title), html_lib.escape(title), html_lib.escape(text)))
    return Response(body, status=status, mimetype="text/html")


@app.get("/")
def home():
    if "html" not in _page:
        try:
            raw = (BASE / "dashboard.html").read_text(encoding="utf-8")
            results = load_results()
        except FileNotFoundError:
            return _plain_page("FloodResQ AI", "dashboard.html is missing on the server. Upload it to the GitHub repository next to app.py.", 503)
        except ResultsError as e:
            return _plain_page("FloodResQ AI", "The map cannot be shown: %s" % e, 503)
        raw, notes, mism = sync_dashboard(raw, results)
        if mism:
            log.warning("dashboard/results mismatch: %s", "; ".join(mism))
        _page["html"] = add_copilot_widget(raw, notes)
        _page["mismatches"] = mism
    return Response(_page["html"], mimetype="text/html")


@app.get("/healthz")
def healthz():
    return jsonify(status="ok")


@app.get("/api/status")
def status():
    """Small JSON for checking the deployment: which files exist, whether the data is consistent, whether a key is set."""
    info = {"dashboard_html": (BASE / "dashboard.html").exists(), "results_json": (BASE / "results.json").exists(),
            "copilot_widget_html": (BASE / "copilot_widget.html").exists(), "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            "gemini_sdk_installed": genai is not None, "model": MODEL, "software": fr.__version__}
    try:
        res = load_results()
        info["verdict"] = (res.get("evidence") or {}).get("verdict")
        info["schema_problems"] = fr.validate_results(res)
    except ResultsError as e:
        info["results_error"] = str(e)
    if "html" not in _page and info["dashboard_html"] and "results_error" not in info:
        home()
    info["dashboard_results_mismatch"] = _page.get("mismatches")
    return jsonify(info)


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
