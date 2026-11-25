"""
The chatbot: uses the Naive Bayes model (naive_bayes.py) to figure out which
topic a question is about, then answers with the pre-written answer for that
topic from chatbot_data.json, filled in with this case's details.

Every answer is written ahead of time, so the chatbot can never make up
medical advice.
"""
import json
import os

from naive_bayes import NaiveBayes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "chatbot_data.json")

# If the model is less sure than this, the chatbot says it isn't sure instead
# of guessing. test_chatbot.py helps you pick this number.
CONFIDENCE_CUTOFF = 0.5
MAX_QUESTION_WORDS = 40   # very long questions make the numbers too tiny to compare

HEAT_STROKE_REMINDER = "Reminder: call 911 and keep cooling the person. "


def load_data(path=DATA_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def training_examples(data):
    """Turns the JSON into two matching lists: questions and their topics."""
    questions, topics = [], []
    for topic, info in data.items():
        for question in info["train"]:
            questions.append(question)
            topics.append(topic)
    return questions, topics


def build_model(data):
    model = NaiveBayes()
    model.train(*training_examples(data))
    return model


DATA = load_data()
MODEL = build_model(DATA)


class _KeepMissing(dict):
    """If an answer uses a {blank} we don't have, leave it alone instead of crashing."""
    def __missing__(self, key):
        return "{" + key + "}"


def fill_in(answer, result):
    """Replaces {blanks} in an answer with this case's details."""
    try:
        temperature_f = f"{float(result.get('temperature_c')) * 9 / 5 + 32:.1f}"
    except (TypeError, ValueError):
        temperature_f = "?"
    details = {
        "label": result.get("label", ""),
        "cns_reason": result.get("cnsReason", "").rstrip("."),
        "heat_reason": result.get("heatReason", "").rstrip("."),
        "temperature_c": result.get("temperature_c", "?"),
        "temperature_f": temperature_f,
        "temp_source": result.get("temp_source") or "not specified",
    }
    return answer.format_map(_KeepMissing(details))


def suggestions(data=DATA):
    """Button labels the page shows so people know what they can ask."""
    return [info["button"] for info in data.values() if info.get("button")]


def reply(question, result):
    """question: what the user typed.
    result: this case's classification (label, reasons, temperature, method).
    Returns the answer plus the topic and confidence, so the page can show them."""
    words = question.split()
    if len(words) > MAX_QUESTION_WORDS:
        question = " ".join(words[:MAX_QUESTION_WORDS])

    topic, confidence = MODEL.predict(question)

    if confidence < CONFIDENCE_CUTOFF:
        answer = "I'm not sure I understood that. Try one of the questions below."
        topic = None
    else:
        info = DATA[topic]
        label = result.get("label", "")
        template = info.get("answers_by_label", {}).get(label, info.get("answer", ""))
        answer = fill_in(template, result)

    if result.get("label") == "Heat Stroke":
        answer = HEAT_STROKE_REMINDER + answer

    return {"answer": answer, "topic": topic, "confidence": round(confidence, 3)}


if __name__ == "__main__":
    # Try it in the terminal:  py chatbot.py
    fake_result = {"label": "Heat Stroke", "cnsReason": "The runner is confused about the day.",
                   "heatReason": "She is sweating heavily.", "temperature_c": 39.6, "temp_source": "oral"}
    print("Ask a question (or press Enter to quit). Pretend result: Heat Stroke\n")
    while True:
        q = input("You: ").strip()
        if not q:
            break
        r = reply(q, fake_result)
        print(f"Bot ({r['topic']}, {r['confidence']:.0%} sure): {r['answer']}\n")
