"""Independent-labeler agreement (review item 2).

IN PLAIN WORDS: the "correct answers" for the vignettes were written by the
student. If they are wrong, every accuracy number is wrong too. An expert
labeling the same cases without seeing the student's answers, plus Cohen's
kappa (agreement beyond luck), shows the answer key is trustworthy.
(Picture: docs/images/08_cohens_kappa.svg)

Circularity problem: the vignettes, their labels, and the prompts all came
from the same person. Fix: get an athletic trainer, school nurse, or
physician to label the vignettes WITHOUT seeing your labels, then measure
how often you agree (Cohen's kappa).

Steps:
  1. python kappa_agreement.py --make-sheet   -> rater_sheet.csv (scenarios + temperature, NO labels)
  2. The rater fills in the 'rater_label' column using exactly one of:
        Heat Stroke | Heat Exhaustion | Uncertain | Not Heat Illness
     (give them the rule's definitions, but do not show them your labels)
  3. python kappa_agreement.py --rater rater_sheet.csv

Note: Heat_Illness_Vignettes_for_Review.docx shows the proposed label next
to each case, which is good for realism feedback but NOT independent
labeling. Use the sheet from step 1 for the independent labels.
"""
import argparse
import csv

from vignette_sets import load_set
from stats_utils import cohens_kappa

SHORT = {
    "Heat Stroke": "Heat Stroke",
    "Heat Exhaustion": "Heat Exhaustion",
    "Uncertain — flag for review": "Uncertain",
    "Not Heat Illness — no thermal stress indicated": "Not Heat Illness",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="all50")
    p.add_argument("--make-sheet", action="store_true")
    p.add_argument("--rater", help="CSV the rater filled in")
    args = p.parse_args()
    vignettes = load_set(args.set)

    if args.make_sheet:
        with open("rater_sheet.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "scenario", "temperature_c", "temperature_method", "rater_label"])
            for v in vignettes:
                w.writerow([v["id"], v["symptom_text"], v["temperature_c"], v.get("temperature_source", ""), ""])
        print("Wrote rater_sheet.csv (no answers included).")
        return

    if not args.rater:
        p.error("use --make-sheet or --rater FILE")
    mine = {v["id"]: SHORT[v["true_label"]] for v in vignettes}
    with open(args.rater, encoding="utf-8", newline="") as f:
        theirs = {r["id"]: r["rater_label"].strip() for r in csv.DictReader(f) if r["rater_label"].strip()}
    ids = [i for i in mine if i in theirs]
    bad = [i for i in ids if theirs[i] not in set(SHORT.values())]
    if bad:
        raise SystemExit(f"Unrecognized rater labels for {bad}. Use exactly: {', '.join(sorted(set(SHORT.values())))}")
    a, b = [mine[i] for i in ids], [theirs[i] for i in ids]
    agree = sum(x == y for x, y in zip(a, b))
    print(f"Cases labeled by rater: {len(ids)}")
    print(f"Raw agreement: {agree}/{len(ids)} = {agree / len(ids) * 100:.1f}%")
    print(f"Cohen's kappa: {cohens_kappa(a, b):.3f}   (0.61-0.80 substantial, 0.81+ almost perfect)")
    print("\nDisagreements (look at these; some mean the vignette or the rule needs fixing):")
    for i in ids:
        if mine[i] != theirs[i]:
            print(f"  {i}: yours = {mine[i]:<16} rater = {theirs[i]}")


if __name__ == "__main__":
    main()
