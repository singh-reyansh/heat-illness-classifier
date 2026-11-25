"""Second baseline (review item 1): a plain keyword search, no AI.

IN PLAIN WORDS: instead of asking an AI "is there a CNS sign?", this just
searches the text for words like "confus" or "collaps" (word stems, so
"confused" and "confusion" both match). It then uses the SAME rule as the AI
pipeline. This is the fair comparison: same rule, only the "reading" differs.
On all 50 vignettes it scored 39/50 (78%); the AI scored 49/50 (98%). It
misses descriptions like "says the sun is too loud" that mean confusion but
don't use any of the listed words.

A temperature-only baseline can never see CNS signs, so it was always going
to lose. The fair question is: does the AI beat a simple word search that
CAN read the text? This script answers that.

Fairness rules for this file:
  * The word lists were written from the NATA/ACSM wording and obvious
    synonyms, BEFORE looking at any scores. Do not add or remove words after
    seeing results; if you want a different list, make a new named variant
    and report both.
  * It uses the exact same decision rule (rule.apply_rule) as the AI
    pipeline, so the only thing that differs is how the two flags are found.
  * Two variants are reported: "naive" (any keyword counts) and
    "negation-aware" (a keyword shortly after words like 'not', 'no',
    "n't", 'without' is ignored, until a comma ends the phrase -- so
    'not confused' no longer counts, but 'not confused, just sweating'
    still counts the sweating).
"""
import argparse
import re

from rule import apply_rule
from vignette_sets import load_set, report_classifier

CNS_KEYWORDS = [
    "confus", "disorient", "delirious", "delirium", "irrational", "combative", "irritab",
    "stumbl", "staggering", "stagger", "lose her balance", "losing her balance", "lose his balance", "losing his balance",
    "loss of balance", "unsteady", "off balance", "collaps", "unresponsive", "unconscious", "passed out", "fainted",
    "seizure", "convuls", "coma", "slurr", "incoherent", "doesn't make sense", "does not make sense", "didn't make sense",
    "out of it", "not make sense", "can't remember", "cannot remember", "no idea where",
]
HEAT_KEYWORDS = [
    "sweat", "weak", "nause", "nauseat", "dizz", "lightheaded", "light-headed", "cramp", "fatigue", "tired", "exhaust",
    "sick to his stomach", "sick to her stomach",
]
NEGATORS = r"(?:not|no|never|without|denies|isn't|wasn't|aren't|doesn't|didn't|n't|nothing)"


def matched_word(text, keywords, negation_aware):
    """Returns the word in the text that made the flag 1 (for example
    "irritable" for the keyword "irritab"), or None if no keyword counts.
    The web page shows this word so judges can see WHY the search fired."""
    lowered = text.lower()
    for kw in keywords:
        for m in re.finditer(re.escape(kw), lowered):
            if negation_aware:
                # Look at the 25 characters just before the keyword. If a
                # "not"-type word is there ("is NOT confused"), skip this match.
                window = lowered[max(0, m.start() - 25):m.start()]
                if re.search(r"\b" + NEGATORS + r"\b[\w\s'-]{0,20}$", window) or re.search(NEGATORS + r"\s*$", window):
                    continue
            start, end = m.start(), m.end()
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            return text[start:end]
    return None


def _found(text, keywords, negation_aware):
    """Returns 1 if any keyword appears in the text, otherwise 0."""
    return 0 if matched_word(text, keywords, negation_aware) is None else 1


def classify_keyword(symptom_text, temperature_c, negation_aware=False):
    cns = _found(symptom_text, CNS_KEYWORDS, negation_aware)
    heat = _found(symptom_text, HEAT_KEYWORDS, negation_aware)
    return apply_rule(cns, heat, temperature_c), cns, heat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="test20", help="test20 (default), extra30, or all50")
    args = parser.parse_args()
    vignettes = load_set(args.set)
    for name, neg in (("keyword_naive", False), ("keyword_negation_aware", True)):
        rows = []
        for v in vignettes:
            predicted, cns, heat = classify_keyword(v["symptom_text"], v["temperature_c"], neg)
            rows.append({"id": v["id"], "predicted_label": predicted, "true_label": v["true_label"],
                         "cns_flag": cns, "heat_symptoms_flag": heat, "temperature_c": v["temperature_c"]})
        report_classifier(name, rows, f"{name}_results_{args.set}.csv")


if __name__ == "__main__":
    main()
