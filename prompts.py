"""The two yes/no questions the AI is asked, in two versions.

IN PLAIN WORDS: a "prompt" is the exact message sent to the AI. Each prompt
has 3 parts glued together by build_prompt():
    PREFIX   "You are assisting a heat-illness triage tool..."   (sets the job)
    QUESTION the CNS question or the heat-symptom question       (the yes/no question)
    SUFFIX   "Respond in exactly this format: Answer: 1 or 0..."  (forces an answer
             format a program can read, so we can pull out the 1 or 0)
Asking two narrow yes/no questions is much more reliable than asking the AI
"what is the diagnosis?", and it lets the fixed rule in rule.py decide.

v1_original : the wording used for the first 20-vignette evaluation.
v2_expanded : review item 16 -- the CNS list now also covers collapse,
              seizure, irritability and delirium, and the prompts say how to
              separate "dizziness" (a heat symptom) from "loss of balance"
              (a CNS sign).

IMPORTANT (test-set hygiene): choose the version BEFORE looking at results,
write it down, and do not edit these strings after seeing a score. If you
need to experiment with wording, use data/dev_cases.json, never the test
vignettes. Keep the same text in classifier.html (check_parity.py verifies
the two match).
"""

PROMPT_VERSIONS = {
    "v1_original": {
        "cns": "Based only on the description below, does it show signs of central nervous system (CNS) dysfunction — confusion, loss of balance, irrational behavior, or coma?",
        "heat": "Based only on the description below, does it mention any heat-related symptom — heavy sweating, weakness, nausea, dizziness, cramping, or fatigue?",
    },
    "v2_expanded": {
        "cns": (
            "Based only on the description below, does it show signs of central nervous system (CNS) dysfunction — "
            "confusion or disorientation, loss of balance or coordination, irrational, irritable, or combative behavior, "
            "delirium, collapse or loss of consciousness, seizure, or coma? "
            "Dizziness or lightheadedness on its own is NOT a CNS sign (it is a heat symptom), but stumbling, being unable "
            "to walk straight, or not making sense IS a CNS sign."
        ),
        "heat": (
            "Based only on the description below, does it mention any heat-related symptom — heavy sweating, weakness, nausea, "
            "dizziness or lightheadedness, cramping, or fatigue? "
            "Do not count confusion, stumbling, loss of balance, or odd behavior here; those are CNS signs and are judged separately."
        ),
    },
}

DEFAULT_PROMPT_VERSION = "v2_expanded"

PREFIX = "You are assisting a heat-illness triage tool, not making a medical diagnosis. "
SUFFIX = "Respond in exactly this format:\nAnswer: 1 or 0\nReason: one sentence"


def build_prompt(kind, symptom_text, context_text="", version=DEFAULT_PROMPT_VERSION):
    """kind is 'cns' or 'heat'. Matches buildPrompt() in classifier.html."""
    question = PROMPT_VERSIONS[version][kind]
    context_line = (
        f"\nAdditional context (what the patient was doing beforehand): {context_text.strip()}\n"
        if (context_text or "").strip() else ""
    )
    return f"{PREFIX}{question}\n\nDescription: {symptom_text}\n{context_line}\n{SUFFIX}"


# Third question, used only when CNS signs were found AND the user typed a
# "what were they doing" note (review item 4): does the note mention exertion
# or heat exposure?
CONTEXT_PROMPT = (
    "You are assisting a heat-illness triage tool, not making a medical diagnosis. "
    "Based only on the note below about what the patient was doing beforehand, does it mention hard physical exertion "
    "or exposure to a hot environment?\n\nNote: {context_text}\n\n" + SUFFIX
)


def build_context_prompt(context_text):
    return CONTEXT_PROMPT.format(context_text=context_text.strip())
