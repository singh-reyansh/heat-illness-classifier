"""
Flask web server for the heat illness classifier.

Why this exists:
  * The Groq API key stays on this computer (read from the GROQ_API_KEY
    environment variable). It is never sent to the browser.
  * The web page uses the SAME Python classifier as the evaluation
    (full_classifier_groq.classify_detailed), so the app judges see is the
    exact code that was tested.

Run (Windows, from this folder):
    set GROQ_API_KEY=your_key
    py server.py
Then open http://localhost:5000 in your browser.
"""
import math
import os
import sys

import requests
from flask import Flask, jsonify, request, send_from_directory

import baseline_classifier
import chatbot
import full_classifier_groq as classifier
import keyword_classifier

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = "classifier_flask.html"  # lives in the web/ folder
MAX_TEXT_LENGTH = 2000  # stops someone pasting a novel and burning API quota

app = Flask(__name__)

# IN PLAIN WORDS (picture: docs/images/09_web_app.svg)
# A web server waits for the browser to ask for things. Each "route" below
# is one address the browser can ask:
#   GET  /                       -> sends the web page
#   GET  /api/status             -> "is the API key set?"
#   POST /api/classify           -> runs the AI classifier on a description
#   POST /api/baselines          -> runs the two no-AI baselines on it
#   GET  /api/chat/suggestions   -> the chatbot's question buttons
#   POST /api/chat               -> asks the Naive Bayes chatbot a question
# GET means "give me something"; POST means "here is some data, work on it".
# The answers go back as JSON, a text format both Python and JavaScript read.


@app.get("/")
def index():
    return send_from_directory(os.path.join(HERE, "web"), PAGE)


@app.get("/api/status")
def status():
    return jsonify(key_set=bool(os.environ.get("GROQ_API_KEY")))


def read_case(data):
    """Checks the form fields. Returns (case, None) or (None, error message)."""
    symptom_text = str(data.get("symptom_text", "")).strip()
    context_text = str(data.get("context_text", "")).strip()
    temp_source = str(data.get("temp_source", "")).strip()
    try:
        temperature_c = float(data.get("temperature_c"))
    except (TypeError, ValueError):
        return None, "Temperature must be a number."
    if not math.isfinite(temperature_c):
        return None, "Temperature must be a number."
    if not symptom_text:
        return None, "Please enter a symptom description."
    if len(symptom_text) > MAX_TEXT_LENGTH or len(context_text) > MAX_TEXT_LENGTH:
        return None, f"Descriptions must be under {MAX_TEXT_LENGTH} characters."
    return (symptom_text, temperature_c, temp_source, context_text), None


@app.post("/api/baselines")
def baselines():
    """The two no-AI baselines from the experiment, run on whatever was typed.
    Uses the exact Python code that was tested (baseline_classifier.py and
    keyword_classifier.py). No API key or internet needed."""
    case, error = read_case(request.get_json(silent=True) or {})
    if error:
        return jsonify(error=error), 400
    symptom_text, temperature_c = case[0], case[1]
    out = {"temperature_only": {"label": baseline_classifier.classify_baseline(temperature_c)}}
    for name, negation_aware in (("keyword_naive", False), ("keyword_negation_aware", True)):
        label, cns, heat = keyword_classifier.classify_keyword(symptom_text, temperature_c, negation_aware)
        out[name] = {
            "label": label, "cnsFlag": cns, "heatFlag": heat,
            "cnsWord": keyword_classifier.matched_word(symptom_text, keyword_classifier.CNS_KEYWORDS, negation_aware),
            "heatWord": keyword_classifier.matched_word(symptom_text, keyword_classifier.HEAT_KEYWORDS, negation_aware),
        }
    return jsonify(out)


@app.post("/api/classify")
def classify():
    case, error = read_case(request.get_json(silent=True) or {})
    if error:
        return jsonify(error=error), 400
    symptom_text, temperature_c, temp_source, context_text = case
    if not os.environ.get("GROQ_API_KEY"):
        return jsonify(error="The server has no Groq API key. Stop the server, run  set GROQ_API_KEY=your_key  and start it again."), 500

    classifier.CALL_LOG.clear()
    try:
        result = classifier.classify_detailed(symptom_text, temperature_c, temp_source, context_text)
    except requests.RequestException as err:
        return jsonify(error=f"The Groq API request failed: {err}"), 502
    except RuntimeError as err:
        return jsonify(error=str(err)), 502

    if result["unparseable"]:
        return jsonify(error="The AI's reply could not be read after 3 tries. Please try again."), 502

    model = classifier.CALL_LOG[-1]["model"] if classifier.CALL_LOG else None
    return jsonify(
        label=result["label"],
        cnsFlag=result["cns_flag"],
        cnsReason=result["cns_reason"].strip(),
        heatFlag=result["heat_flag"],
        heatReason=result["heat_reason"].strip(),
        caveats=result["caveats"],
        model=model,
    )


@app.get("/api/chat/suggestions")
def chat_suggestions():
    return jsonify(suggestions=chatbot.suggestions())


@app.post("/api/chat")
def chat():
    """The Naive Bayes chatbot. Runs entirely on this computer: no API key,
    no internet."""
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    result = data.get("result") or {}
    if not question:
        return jsonify(error="Please type a question."), 400
    if len(question) > 300:
        return jsonify(error="Please keep questions under 300 characters."), 400
    if not isinstance(result, dict):
        return jsonify(error="Missing result."), 400
    # Only keep the fields the chatbot uses, as short plain strings.
    safe_result = {key: str(result.get(key, ""))[:500]
                   for key in ("label", "cnsReason", "heatReason", "temperature_c", "temp_source")}
    return jsonify(chatbot.reply(question, safe_result))


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set. Live classification will fail (Demo mode still works).")
    # Port 5000 by default. "py server.py --port 80" lets the address drop the
    # ":5000" part. Any name ending in .localhost goes to this computer, so the
    # page also opens at http://heatstrokeclassifier.localhost
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 5000
    suffix = "" if port == 80 else f":{port}"
    print(f"Open http://heatstrokeclassifier.localhost{suffix} in your browser. Press Ctrl+C to stop.")
    if "--phone" in sys.argv:
        # --phone: phones and tablets on the SAME Wi-Fi can open the page too,
        # using this laptop's Wi-Fi address (a phone doesn't know the name
        # "heatstrokeclassifier"). Anyone on that Wi-Fi could use it, so only
        # turn this on for a demo, on a network you trust.
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))  # picks the Wi-Fi address; nothing is sent
                lan_ip = s.getsockname()[0]
        except OSError:
            lan_ip = "this laptop's Wi-Fi address"
        print(f"On a phone connected to the same Wi-Fi, open:  http://{lan_ip}{suffix}")
        print("If Windows asks whether to allow Python on networks, choose Private networks and click Allow.")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        # 127.0.0.1 = only this computer can open the page.
        app.run(host="127.0.0.1", port=port, debug=False)
