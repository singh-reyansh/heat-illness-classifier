"""
Makes the accuracy bar chart for the poster, from the real results files.

IN PLAIN WORDS: reads the results CSV of every classifier (made by running
full_classifier_groq.py, keyword_classifier.py and baseline_classifier.py),
works out each one's accuracy and 95% Wilson range, and draws a bar chart
with error bars. The error bar is the Wilson range: the TRUE accuracy is
probably somewhere inside it.

Run:  py make_chart.py --set all50      -> saves results/accuracy_chart_all50.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # draw straight to a file, no window
import matplotlib.pyplot as plt

from stats_utils import wilson_confidence_interval

HERE = os.path.dirname(os.path.abspath(__file__))

# Which results file belongs to which bar, and its color (left to right).
CLASSIFIERS = [
    ("Temperature\nonly", "baseline_classifier_results_{set}.csv", "#9a9a94"),
    ("Keyword\nsearch", "keyword_negation_aware_results_{set}.csv", "#2a78d6"),
    ("AI pipeline\n(LLM + NATA rule)", "full_classifier_groq_results_{set}.csv", "#eb6834"),
]

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"


def accuracy_from_csv(path):
    """Returns (number correct, number of cases) from a results CSV."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    correct = sum(r["predicted_label"] == r["true_label"] for r in rows)
    return correct, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="all50", help="test20, extra30, or all50 (default)")
    args = parser.parse_args()

    names, accuracy, err_low, err_high, colors = [], [], [], [], []
    n_cases = None
    for name, pattern, color in CLASSIFIERS:
        path = os.path.join(HERE, "results", pattern.format(set=args.set))
        if not os.path.exists(path):
            print(f"(skipping {name.replace(chr(10), ' ')}: {os.path.basename(path)} not found, run that script first)")
            continue
        correct, n = accuracy_from_csv(path)
        n_cases = n
        lo, hi = wilson_confidence_interval(correct, n)
        acc = correct / n * 100
        names.append(name)
        accuracy.append(acc)
        err_low.append(acc - lo * 100)   # error bars are drawn as distances from the bar's top
        err_high.append(hi * 100 - acc)
        colors.append(color)
        print(f"{name.replace(chr(10), ' '):<32} {correct}/{n} = {acc:.1f}%   (95% Wilson range {lo * 100:.1f}% to {hi * 100:.1f}%)")
    if not names:
        raise SystemExit("No results files found. Run the classifier scripts first.")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = list(range(len(names)))
    ax.bar(x, accuracy, width=0.55, color=colors, zorder=3)
    ax.errorbar(x, accuracy, yerr=[err_low, err_high], fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1.5, capsize=8, capthick=1.5, zorder=4)

    # The accuracy number above each bar (above the error bar's cap)
    for xi, a, eh in zip(x, accuracy, err_high):
        ax.text(xi, a + eh + 3, f"{a:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK_PRIMARY)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{t}%" for t in [0, 20, 40, 60, 80, 100]], color=INK_MUTED, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=INK_PRIMARY, fontsize=11)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", length=0)

    ax.set_title("Classification accuracy on the same vignettes", fontsize=14, fontweight="bold", color=INK_PRIMARY, pad=18)
    fig.text(0.5, 0.01, f"n = {n_cases} vignettes ({args.set})  •  error bars = 95% Wilson confidence interval",
             ha="center", fontsize=9.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(HERE, "results", f"accuracy_chart_{args.set}.png")
    plt.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved results/{os.path.basename(out)}")


if __name__ == "__main__":
    main()
