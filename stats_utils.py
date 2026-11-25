"""
STATISTICS TOOLBOX -- every "how good is it, and how sure are we?" calculation.

IN PLAIN WORDS
  * wilson_confidence_interval : a range for the TRUE accuracy, not just the
                                 measured one (picture: docs/images/05_wilson_interval.svg)
  * confusion_matrix_report    : a table of where right and wrong answers went,
                                 plus precision and recall (docs/images/04_confusion_matrix.svg)
  * mcnemar_exact              : is classifier A really better than B, or just
                                 lucky? (docs/images/06_mcnemar_test.svg)
  * cohens_kappa               : do two people agree more than by luck?
                                 (docs/images/08_cohens_kappa.svg)
  * heat_stroke_summary        : how many real heat strokes were caught or missed
  * consistency_report         : does the AI give the same answer every run?

Everything is written by hand with only Python's math module, so you can see
exactly how each number is calculated.
"""
import math


def wilson_confidence_interval(correct, n, z=1.96):
    """Computes the Wilson score confidence interval for a proportion
    (here, classification accuracy).

    Why not just report accuracy +/- a simple margin? Because that simple
    approach breaks down exactly when it matters most for this project:
    with a small n, and especially when accuracy is close to 0% or 100%
    (like an 8/8 or 19/20 result), a plain +/- margin can produce a range
    that goes below 0% or above 100%, which is meaningless. The Wilson
    interval doesn't have that problem and is the standard fix for small
    or extreme-proportion samples.

    correct: number of correct predictions
    n: total number of predictions
    z: z-score for the desired confidence level (1.96 = 95% confidence,
       meaning: if you repeated this whole evaluation many times, about
       95% of the intervals you'd compute this way would contain the
       system's true, long-run accuracy)

    Returns (lower_bound, upper_bound) as fractions between 0 and 1.
    Multiply by 100 to report as a percentage.
    """
    # IN PLAIN WORDS: 49 right out of 50 is 98%, but with only 50 cases we
    # can't be sure the tool would score exactly 98% on every possible case.
    # This returns a range (for 49/50: about 89.5% to 99.6%) that the true,
    # long-run accuracy is very likely inside.
    #
    # Worked example with correct=49, n=50, z=1.96 (z squared = 3.8416):
    #   phat        = 49/50                                    = 0.98
    #   denominator = 1 + 3.8416/50                            = 1.0768
    #   center      = 0.98 + 3.8416/100                        = 1.0184
    #   adjustment  = 1.96 * sqrt(0.98*0.02/50 + 3.8416/10000) = 0.0546
    #   lower = (1.0184 - 0.0546) / 1.0768 = 0.895  ->  89.5%
    #   upper = (1.0184 + 0.0546) / 1.0768 = 0.996  ->  99.6%
    if n == 0:
        return 0.0, 0.0

    phat = correct / n                       # the measured accuracy ("p hat")
    denominator = 1 + z**2 / n               # shrinks the range a bit for small samples
    center = phat + z**2 / (2 * n)           # nudges the middle toward 50%, so the range can't pass 0% or 100%
    adjustment = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))  # how far the range spreads

    lower = (center - adjustment) / denominator
    upper = (center + adjustment) / denominator

    # Clamp to [0, 1] to guard against tiny floating-point overshoot
    return max(0.0, lower), min(1.0, upper)


def confusion_matrix_report(true_labels, predicted_labels):
    """Builds and prints a confusion matrix, plus per-class precision,
    recall, and F1 score. Written from scratch instead of using
    scikit-learn so you don't need to install another package -- it's a
    small enough calculation to do by hand.

    true_labels / predicted_labels: two parallel lists of label strings
    (same length, same order -- true_labels[i] and predicted_labels[i]
    refer to the same vignette).

    Precision for a label answers: "of everything the pipeline called
    this label, how much actually was?" Recall answers: "of everything
    that actually was this label, how much did the pipeline catch?" A
    label with low recall is one the pipeline tends to miss; a label
    with low precision is one the pipeline over-uses.
    """
    labels = sorted(set(true_labels) | set(predicted_labels))

    # confusion[true_label][predicted_label] = how many times that happened
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(true_labels, predicted_labels):
        confusion[t][p] += 1

    # Shorten long label names so the matrix fits on screen
    short = {l: (l[:16] + "..") if len(l) > 18 else l for l in labels}
    col_width = 20

    print("Confusion matrix (rows = true label, columns = what the pipeline predicted):")
    header = " " * col_width + "".join(f"{short[l]:<{col_width}}" for l in labels)
    print(header)
    for t in labels:
        row = f"{short[t]:<{col_width}}" + "".join(
            f"{confusion[t][p]:<{col_width}}" for p in labels
        )
        print(row)
    print()

    print(f"{'Label':<45}{'Precision':<12}{'Recall':<10}{'F1':<8}{'Support'}")
    print("-" * 85)
    for l in labels:
        tp = confusion[l][l]
        fn = sum(confusion[l][p] for p in labels if p != l)  # true=l, predicted wrong
        fp = sum(confusion[t][l] for t in labels if t != l)  # predicted=l, actually wrong
        support = tp + fn  # how many vignettes were actually this label

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"{l:<45}{precision * 100:>8.1f}%   {recall * 100:>6.1f}%   {f1 * 100:>5.1f}%  {support}")


# ---------------------------------------------------------------------------
# Added after expert review
# ---------------------------------------------------------------------------
def mcnemar_exact(a_correct, b_correct):
    """McNemar's exact test for two classifiers run on the SAME cases.

    a_correct, b_correct: parallel lists of True/False (did classifier A /
    B get case i right?). Only the cases where they DISAGREE carry
    information:
        only_a = A right, B wrong        only_b = A wrong, B right
    Under "no real difference", each disagreement is equally likely to go
    either way, so only_a ~ Binomial(only_a + only_b, 0.5). The exact
    two-sided p-value is computed directly from that (no approximation, so
    it is fine for small samples like 20 or 50 cases).

    Returns a dict with only_a, only_b, both_right, both_wrong, p_value.

    IN PLAIN WORDS: imagine both classifiers took the same 50-question test.
    Questions both got right (or both got wrong) can't tell us who is better,
    so they are ignored. On the questions where they disagree, if the two were
    equally good, each disagreement would be a 50/50 coin flip. The p-value is
    the chance of a result at least this lopsided from fair coin flips.

    Real example (AI vs keyword search on all50): they disagreed on 10 cases
    and the AI won all 10. Ten heads in a row = (1/2)^10 = 1/1024. Doubled,
    because 10-to-0 the OTHER way would be just as surprising ("two-sided"):
    p = 2/1024 = 0.002. Below 0.05 -> very unlikely to be luck.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("The two result lists must be the same length (same cases, same order).")
    only_a = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    only_b = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    both_right = sum(1 for x, y in zip(a_correct, b_correct) if x and y)
    both_wrong = sum(1 for x, y in zip(a_correct, b_correct) if not x and not y)
    n = only_a + only_b          # number of disagreements = number of "coin flips"
    if n == 0:
        p = 1.0                  # they never disagreed: no evidence either way
    else:
        k = min(only_a, only_b)  # the smaller side's wins
        # math.comb(n, i) = how many ways to get exactly i "heads" in n flips.
        # Adding up i = 0..k and dividing by 2**n (all possible outcomes) gives
        # the chance of k or fewer wins for the smaller side. Times 2 = two-sided.
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "both_right": both_right, "both_wrong": both_wrong, "p_value": p}


def cohens_kappa(labels_1, labels_2):
    """Agreement between two labelers, corrected for chance.
    1.0 = perfect agreement, 0 = no better than chance. Rough guide:
    0.61-0.80 substantial, 0.81-1.00 almost perfect.

    IN PLAIN WORDS: two people who just guessed would still agree sometimes
    by luck. Kappa asks how much of the agreement is MORE than luck:
        kappa = (observed agreement - chance agreement) / (1 - chance agreement)
    Example: agree on 90% of cases, 30% expected by chance ->
        (0.90 - 0.30) / (1 - 0.30) = 0.86 (almost perfect)."""
    if len(labels_1) != len(labels_2) or not labels_1:
        raise ValueError("Need two equal-length, non-empty label lists.")
    n = len(labels_1)
    observed = sum(a == b for a, b in zip(labels_1, labels_2)) / n   # fraction of cases where they matched
    classes = set(labels_1) | set(labels_2)
    # Chance agreement: for each label, (how often person 1 uses it) x (how
    # often person 2 uses it) = the chance they would both pick it at random.
    expected = sum((labels_1.count(c) / n) * (labels_2.count(c) / n) for c in classes)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def heat_stroke_summary(true_labels, predicted_labels):
    """Heat stroke is the label where a miss is dangerous, so report it alone.

    strict_recall      : true Heat Stroke cases the tool called Heat Stroke.
    dangerous_misses   : true Heat Stroke called Heat Exhaustion, Not Heat
                         Illness, or unparseable. A flag for review does NOT
                         count as dangerous (a person then looks at it).
    """
    hs = "Heat Stroke"
    pairs = [(t, p) for t, p in zip(true_labels, predicted_labels) if t == hs]
    n_true = len(pairs)
    strict_hits = sum(1 for _, p in pairs if p == hs)
    flagged = sum(1 for _, p in pairs if p.startswith("Uncertain"))
    dangerous = n_true - strict_hits - flagged
    return {
        "n_true": n_true,
        "strict_hits": strict_hits,
        "strict_recall": (strict_hits / n_true) if n_true else float("nan"),
        "flagged_for_review": flagged,
        "dangerous_misses": dangerous,
    }


def consistency_report(runs_predictions):
    """runs_predictions: list of runs, each a list of predicted labels (same
    case order). Returns the fraction of cases that got the SAME label in
    every run, plus the majority-vote label for each case.

    IN PLAIN WORDS: an AI can give slightly different answers to the same
    question. Running every case 3+ times and checking whether the label
    stayed the same measures how repeatable the tool is. When runs disagree,
    the label most runs chose ("majority vote") becomes the final answer."""
    n_cases = len(runs_predictions[0])
    majority, identical = [], 0
    for i in range(n_cases):
        votes = [run[i] for run in runs_predictions]
        top = max(set(votes), key=lambda lab: (votes.count(lab), lab))
        majority.append(top)
        identical += len(set(votes)) == 1
    return {"identical_fraction": identical / n_cases, "identical_cases": identical, "n_cases": n_cases, "majority_labels": majority}
