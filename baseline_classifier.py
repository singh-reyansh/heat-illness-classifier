"""Baseline 1: temperature only. No AI, no text reading.

IN PLAIN WORDS: a "baseline" is a simple method to compare against. If the AI
can't beat a simple method, the AI isn't worth using. This one just checks
the thermometer: above 40.5 C -> Heat Stroke, otherwise Heat Exhaustion.
On all 50 vignettes it scored 24/50 (48%) and caught only 8 of 17 heat
strokes, because many heat strokes have a normal-looking field temperature.

It can only ever say "Heat Stroke" or "Heat Exhaustion", so it can never
detect CNS signs or call a case "Not Heat Illness". That makes it an easy
baseline to beat -- which is why keyword_classifier.py (a baseline that CAN
read the text) is the more meaningful comparison.
"""
import argparse

from rule import is_above_threshold, LABEL_HEAT_STROKE, LABEL_HEAT_EXHAUSTION
from vignette_sets import load_set, report_classifier


def classify_baseline(temperature_c):
    return LABEL_HEAT_STROKE if is_above_threshold(temperature_c) else LABEL_HEAT_EXHAUSTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="test20", help="test20 (default), extra30, or all50")
    args = parser.parse_args()
    rows = []
    for v in load_set(args.set):
        rows.append({"id": v["id"], "temperature_c": v["temperature_c"],
                     "predicted_label": classify_baseline(v["temperature_c"]), "true_label": v["true_label"]})
    report_classifier("baseline_temperature_only", rows, f"baseline_classifier_results_{args.set}.csv")


if __name__ == "__main__":
    main()
