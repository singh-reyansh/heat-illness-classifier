"""Compare the classifiers on the SAME cases (review items 6, 8, 9).

IN PLAIN WORDS: this is where the experiment's answer comes from. It reads
the saved results of every classifier and puts them side by side. Real
all50 results: AI 49/50 (98%), keyword search 39/50 (78%), temperature only
24/50 (48%). McNemar's test then checks whether the AI's lead could be luck
(vs keywords: p = 0.002, so no). Finally it lists every case the AI got
wrong so you can explain WHY -- judges love a good error analysis.

Run the three classifiers first (each saves a results CSV), then:
    python compare_classifiers.py --set test20

What it prints:
  1. Accuracy + 95% Wilson interval for every classifier.
  2. McNemar's exact test: AI-assisted vs each baseline. This is the right
     test because every classifier saw the same cases (paired results). It
     only counts the cases where the two disagree.
  3. Heat Stroke recall and dangerous misses for each classifier.
  4. ERROR ANALYSIS: every case the AI-assisted classifier got wrong, with
     the AI's two yes/no answers and its one-sentence reasons, plus what each
     baseline did on that case -- so you can explain WHY it was wrong.
Nothing here is tuned; it only reads the saved CSV files.
"""
import argparse
import csv
import os

from vignette_sets import RESULTS_DIR
from rule import is_dangerous_miss
from stats_utils import mcnemar_exact, wilson_confidence_interval, heat_stroke_summary

FILES = {
    "AI-assisted": "full_classifier_groq_results_{set}.csv",
    "Keyword (naive)": "keyword_naive_results_{set}.csv",
    "Keyword (negation-aware)": "keyword_negation_aware_results_{set}.csv",
    "Temperature only": "baseline_classifier_results_{set}.csv",
}


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="test20")
    args = parser.parse_args()

    data = {}
    for name, pattern in FILES.items():
        path = os.path.join(RESULTS_DIR, pattern.format(set=args.set))
        if os.path.exists(path):
            data[name] = load(path)
        else:
            print(f"(skipping {name}: {os.path.basename(path)} not found — run that script first)")
    if "AI-assisted" not in data or len(data) < 2:
        raise SystemExit("Need the AI-assisted results plus at least one baseline. Run full_classifier_groq.py and a baseline first.")

    ids = sorted(data["AI-assisted"])
    for name, rows in data.items():
        if sorted(rows) != ids:
            raise SystemExit(f"{name} was run on different cases than the AI-assisted classifier; cannot pair them.")
    true = {i: data["AI-assisted"][i]["true_label"] for i in ids}
    n = len(ids)

    print(f"\nSet: {args.set}   cases: {n}\n")
    print(f"{'Classifier':<28}{'Accuracy':<14}{'95% Wilson CI':<20}{'Heat Stroke recall':<22}{'Dangerous misses'}")
    print("-" * 100)
    correct = {}
    for name, rows in data.items():
        c = [rows[i]["predicted_label"] == true[i] for i in ids]
        correct[name] = c
        lo, hi = wilson_confidence_interval(sum(c), n)
        s = heat_stroke_summary([true[i] for i in ids], [rows[i]["predicted_label"] for i in ids])
        print(f"{name:<28}{sum(c)}/{n} = {sum(c) / n * 100:4.1f}%   {lo * 100:4.1f}% - {hi * 100:4.1f}%     "
              f"{s['strict_hits']}/{s['n_true']}                 {s['dangerous_misses']}")

    print("\nMcNemar's exact test (AI-assisted vs each baseline, paired on the same cases)")
    print("-" * 100)
    for name in data:
        if name == "AI-assisted":
            continue
        r = mcnemar_exact(correct["AI-assisted"], correct[name])
        print(f"vs {name:<26} AI right & baseline wrong: {r['only_a']:<3} AI wrong & baseline right: {r['only_b']:<3} "
              f"both right: {r['both_right']:<3} both wrong: {r['both_wrong']:<3} p = {r['p_value']:.4f}")
    print("(p below 0.05 = the difference is unlikely to be chance. With few cases, even a big gap can fail to reach it —")
    print(" report that honestly; a bigger set like 'all50' gives the test more power.)")

    print("\nERROR ANALYSIS — cases the AI-assisted classifier got wrong")
    print("-" * 100)
    wrong = [i for i in ids if data["AI-assisted"][i]["predicted_label"] != true[i]]
    if not wrong:
        print("None on this set.")
    for i in wrong:
        row = data["AI-assisted"][i]
        print(f"{i}: {row.get('symptom_text', '')}")
        print(f"   temperature: {row.get('temperature_c', '?')} C ({row.get('temperature_source', '') or 'method not given'})")
        print(f"   true label:      {true[i]}")
        print(f"   AI-assisted:     {row['predicted_label']}"
              + ("   <-- DANGEROUS MISS" if is_dangerous_miss(true[i], row['predicted_label']) else ""))
        print(f"   AI said CNS={row.get('cns_flag')} ('{row.get('cns_reason', '')}')")
        print(f"   AI said heat symptom={row.get('heat_symptoms_flag')} ('{row.get('heat_symptoms_reason', '')}')")
        if row.get("labels_each_run") and "|" in row["labels_each_run"]:
            print(f"   label in each run: {row['labels_each_run']}")
        for name, rows in data.items():
            if name != "AI-assisted":
                print(f"   {name}: {rows[i]['predicted_label']}")
        print("   WHY (fill in): was the description ambiguous? did the AI misread it? is the true label itself debatable?\n")


if __name__ == "__main__":
    main()
