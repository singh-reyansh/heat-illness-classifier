"""Bakes the real evaluation results into the web page, so the Results
dashboard, the Mistakes gallery and the method comparison work with no
internet and always show the numbers the experiment actually produced.

IN PLAIN WORDS: compare_classifiers.py prints the results in the terminal.
This script calculates the same numbers with the same code (stats_utils.py)
and writes them INTO the page between the /*RESULTS_START*/ and
/*RESULTS_END*/ markers, the same way build_demo_into_html.py does for demo
mode. Nothing is typed in by hand except the "why" notes in ERROR_NOTES.

    1. Run the classifiers on a set (full_classifier_groq.py, keyword_classifier.py,
       baseline_classifier.py) so their results CSVs exist.
    2. py build_results_into_html.py web/classifier_flask.html                (all50)
       py build_results_into_html.py web/classifier_flask.html --set test20
       py build_results_into_html.py web/classifier_flask.html --rater rater_sheet.csv
       (--rater adds the expert-agreement numbers once the expert has filled in the sheet)

Run it again any time you re-run the evaluation.
"""
import csv
import json
import os
import sys

from compare_classifiers import FILES, load
from kappa_agreement import SHORT
from keyword_classifier import CNS_KEYWORDS, HEAT_KEYWORDS, classify_keyword, matched_word
from rule import ALL_LABELS, is_dangerous_miss
from stats_utils import cohens_kappa, heat_stroke_summary, mcnemar_exact, wilson_confidence_interval
from vignette_sets import RESULTS_DIR, load_set

# The "why" for every case the AI got wrong, in my own words (error analysis).
# Add a note here for any new mistake after re-running the evaluation.
ERROR_NOTES = {
    "V008": ("The prompt lists \"irritable or combative behavior\" as a CNS sign, so the AI followed its "
             "instructions. The real gap is in the rule: it doesn't check for heat or exercise context. "
             "Irritability alone, with a normal temperature and no heat symptoms, may not be heat illness. "
             "This is a false alarm (an unneeded 911 call), not a dangerous miss."),
}

METHOD_KEYS = {
    "AI-assisted": "ai",
    "Keyword (naive)": "keyword_naive",
    "Keyword (negation-aware)": "keyword_negation_aware",
    "Temperature only": "temperature_only",
}
METHOD_NAMES = {
    "ai": "AI pipeline (LLM + NATA rule)",
    "keyword_naive": "Keyword search (no AI)",
    "keyword_negation_aware": "Keyword search, \"not\"-aware (no AI)",
    "temperature_only": "Temperature only (no AI)",
}


def build(set_name, rater_path=None):
    vignettes = {v["id"]: v for v in load_set(set_name)}
    data = {}
    for name, pattern in FILES.items():
        path = os.path.join(RESULTS_DIR, pattern.format(set=set_name))
        if not os.path.exists(path):
            raise SystemExit(f"Missing {os.path.basename(path)}. Run that classifier with --set {set_name} first.")
        data[METHOD_KEYS[name]] = load(path)

    ids = sorted(data["ai"])
    for key, rows in data.items():
        if sorted(rows) != ids:
            raise SystemExit(f"{key} was run on different cases than the AI; cannot pair them.")
    true = {i: data["ai"][i]["true_label"] for i in ids}

    cases = []
    for i in ids:
        ai, v = data["ai"][i], vignettes[i]
        case = {
            "id": i, "text": v["symptom_text"], "temperature_c": v["temperature_c"],
            "source": v.get("temperature_source", ""), "true": true[i],
            "ai": {"label": ai["predicted_label"], "cnsFlag": int(ai["cns_flag"]), "cnsReason": ai["cns_reason"],
                   "heatFlag": int(ai["heat_symptoms_flag"]), "heatReason": ai["heat_symptoms_reason"],
                   "eachRun": ai.get("labels_each_run", "").split("|")},
            "temperature_only": {"label": data["temperature_only"][i]["predicted_label"]},
        }
        for key, negation_aware in (("keyword_naive", False), ("keyword_negation_aware", True)):
            label, cns, heat = classify_keyword(v["symptom_text"], v["temperature_c"], negation_aware)
            if label != data[key][i]["predicted_label"]:
                raise SystemExit(f"{i}: keyword_classifier.py now gives a different answer than {key}_results CSV. Re-run it.")
            case[key] = {"label": label, "cnsFlag": cns, "heatFlag": heat,
                         "cnsWord": matched_word(v["symptom_text"], CNS_KEYWORDS, negation_aware),
                         "heatWord": matched_word(v["symptom_text"], HEAT_KEYWORDS, negation_aware)}
        if ai["predicted_label"] != true[i]:
            case["why"] = ERROR_NOTES.get(i, "")
        cases.append(case)

    n = len(ids)
    correct = {key: [data[key][i]["predicted_label"] == true[i] for i in ids] for key in data}
    methods = []
    for key in ("ai", "keyword_naive", "keyword_negation_aware", "temperature_only"):
        predicted = [data[key][i]["predicted_label"] for i in ids]
        lo, hi = wilson_confidence_interval(sum(correct[key]), n)
        labels = ALL_LABELS + sorted({p for p in predicted if p not in ALL_LABELS})
        matrix = [[sum(1 for i in ids if true[i] == t and data[key][i]["predicted_label"] == p) for p in labels]
                  for t in ALL_LABELS]
        hs = heat_stroke_summary([true[i] for i in ids], predicted)
        methods.append({
            "key": key, "name": METHOD_NAMES[key], "correct": sum(correct[key]), "n": n,
            "wilson": [lo, hi], "hsCaught": hs["strict_hits"], "hsTotal": hs["n_true"],
            "hsFlagged": hs["flagged_for_review"], "dangerousMisses": hs["dangerous_misses"],
            "falseAlarms": sum(1 for i in ids if true[i] != ALL_LABELS[0] and data[key][i]["predicted_label"] == ALL_LABELS[0]),
            "matrixCols": labels, "matrix": matrix,
        })

    mcnemar = {}
    for key in ("keyword_naive", "keyword_negation_aware", "temperature_only"):
        r = mcnemar_exact(correct["ai"], correct[key])
        mcnemar[key] = {"aiOnly": r["only_a"], "baselineOnly": r["only_b"], "bothRight": r["both_right"],
                        "bothWrong": r["both_wrong"], "p": r["p_value"]}

    meta_path = os.path.join(RESULTS_DIR, f"run_metadata_{set_name}.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    runs = meta.get("runs", 1)
    consistency = None
    if runs > 1:
        same = sum(1 for c in cases if len(set(c["ai"]["eachRun"])) == 1)
        consistency = {"identical": same, "n": n}

    kappa = None
    if rater_path:
        mine = {i: SHORT[true[i]] for i in ids}
        with open(rater_path, encoding="utf-8", newline="") as f:
            theirs = {r["id"]: r["rater_label"].strip() for r in csv.DictReader(f) if r["rater_label"].strip()}
        both = [i for i in ids if i in theirs]
        bad = [i for i in both if theirs[i] not in set(SHORT.values())]
        if bad:
            raise SystemExit(f"Unrecognized rater labels for {bad}. Use exactly: {', '.join(sorted(set(SHORT.values())))}")
        if both:
            a, b = [mine[i] for i in both], [theirs[i] for i in both]
            kappa = {"n": len(both), "agree": sum(x == y for x, y in zip(a, b)), "kappa": cohens_kappa(a, b),
                     "disagreements": [{"id": i, "mine": mine[i], "expert": theirs[i]} for i in both if mine[i] != theirs[i]]}

    return {
        "set": set_name, "n": n, "labels": ALL_LABELS,
        "run": {"date": meta.get("run_started_utc", "")[:10], "model": ", ".join(meta.get("models_that_answered", [])),
                "promptVersion": meta.get("prompt_version", ""), "temperature": meta.get("temperature_setting", 0),
                "runs": runs, "apiCalls": meta.get("total_api_calls"), "unparseable": meta.get("unparseable_case_runs")},
        "methods": methods, "mcnemar": mcnemar, "consistency": consistency, "kappa": kappa, "cases": cases,
        "dangerousMissIds": {m: [i for i in ids if is_dangerous_miss(true[i], data[m][i]["predicted_label"])] for m in data},
    }


def main():
    args = sys.argv[1:]
    options = {"--set": "all50", "--rater": None}
    for flag in options:
        if flag in args:
            i = args.index(flag)
            options[flag] = args[i + 1]
            del args[i:i + 2]
    if not args:
        raise SystemExit("usage: py build_results_into_html.py web/classifier_flask.html [--set all50] [--rater rater_sheet.csv]")

    results = build(options["--set"], options["--rater"])
    blob = json.dumps(results, ensure_ascii=False).replace("</", "<\\/")
    for path in args:
        src = open(path, encoding="utf-8").read()
        a, b = src.index("/*RESULTS_START*/"), src.index("/*RESULTS_END*/")
        open(path, "w", encoding="utf-8").write(src[:a] + "/*RESULTS_START*/" + blob + src[b:])
        ai = results["methods"][0]
        print(f"{path}: baked in {results['n']} cases ({results['set']}); AI {ai['correct']}/{ai['n']}"
              + (f"; expert kappa {results['kappa']['kappa']:.2f}" if results["kappa"] else "; no expert labels yet"))


if __name__ == "__main__":
    main()
