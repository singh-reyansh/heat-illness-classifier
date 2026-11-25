"""Loading the vignette sets and printing/saving a classifier's results.

IN PLAIN WORDS: every classifier script uses this file to (1) load a set of
vignettes by name and (2) print the same report: each case, the accuracy with
its Wilson range, heat stroke recall, dangerous misses, and the confusion
matrix, then save everything to a CSV file. Sharing one report function makes
every classifier's results directly comparable. (Picture of the sets:
docs/images/03_data_sets.svg)

test20  : data/vignettes_test20.json  -- the original 20. LOCKED test set. Never use it
          to tune prompts, keyword lists, or anything else.
extra30 : data/vignettes_extra30.json -- 30 more written for expert review
          (V021-V050). Treat as a second, held-out test set.
all50   : both of the above.
dev     : data/dev_cases.json -- practice cases (V101+). The ONLY place to
          experiment with prompt wording.
"""
import csv
import json
import os

from rule import is_dangerous_miss, LABEL_HEAT_STROKE
from stats_utils import confusion_matrix_report, wilson_confidence_interval, heat_stroke_summary

HERE = os.path.dirname(os.path.abspath(__file__))
# Every results file (CSVs, saved AI replies, run records, charts) goes here.
RESULTS_DIR = os.path.join(HERE, "results")
FILES = {
    "test20": [os.path.join("data", "vignettes_test20.json")],
    "extra30": [os.path.join("data", "vignettes_extra30.json")],
    "all50": [os.path.join("data", "vignettes_test20.json"), os.path.join("data", "vignettes_extra30.json")],
    "dev": [os.path.join("data", "dev_cases.json")],
}


def load_set(name):
    if name not in FILES:
        raise SystemExit(f"Unknown set {name!r}. Choose from: {', '.join(FILES)}")
    out = []
    for rel in FILES[name]:
        with open(os.path.join(HERE, rel), encoding="utf-8") as f:
            out.extend(json.load(f))
    return out


def report_classifier(name, rows, csv_filename):
    """rows: dicts with at least id, predicted_label, true_label."""
    n = len(rows)
    correct = sum(r["predicted_label"] == r["true_label"] for r in rows)
    lo, hi = wilson_confidence_interval(correct, n)
    print(f"\n=== {name} ===")
    print(f"{'ID':<6}{'Predicted':<48}{'True label':<48}Match")
    print("-" * 110)
    for r in rows:
        mark = "correct" if r["predicted_label"] == r["true_label"] else "WRONG"
        print(f"{r['id']:<6}{r['predicted_label']:<48}{r['true_label']:<48}{mark}")
    print("-" * 110)
    print(f"Accuracy (strict exact match): {correct}/{n} = {correct / n * 100:.1f}%   95% Wilson CI {lo * 100:.1f}% to {hi * 100:.1f}%")
    s = heat_stroke_summary([r["true_label"] for r in rows], [r["predicted_label"] for r in rows])
    print(f"Heat Stroke recall: {s['strict_hits']}/{s['n_true']} strict"
          + (f" ({s['strict_recall'] * 100:.0f}%)" if s["n_true"] else "")
          + f"; {s['dangerous_misses']} dangerous miss(es) (true Heat Stroke called Heat Exhaustion / Not Heat Illness)"
          + (f": {', '.join(r['id'] for r in rows if is_dangerous_miss(r['true_label'], r['predicted_label']))}" if s["dangerous_misses"] else ""))
    print()
    confusion_matrix_report([r["true_label"] for r in rows], [r["predicted_label"] for r in rows])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, csv_filename), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results/{csv_filename}")
