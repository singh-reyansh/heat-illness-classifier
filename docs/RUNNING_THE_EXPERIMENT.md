# Heat illness classifier — v2 (after expert review)

Run everything from this folder. Windows: use `py` (or `python`) and `set GROQ_API_KEY=your_key`.
Mac/Linux: `export GROQ_API_KEY=your_key`.

## Order that keeps the test honest
1. **Decide the prompt version first** (`prompts.py`: `v1_original` or `v2_expanded`) and write it down. Do not edit prompt text after you have seen a score.
2. If you want to experiment with wording, do it only on the practice cases: `python full_classifier_groq.py --set dev`.
3. Run the locked test set (the original 20), 3+ times to measure consistency:
   `python full_classifier_groq.py --set test20 --runs 3`
4. Run the baselines: `python baseline_classifier.py` and `python keyword_classifier.py`
5. Compare: `python compare_classifiers.py --set test20`  (accuracy, McNemar's test, Heat Stroke recall, error analysis)
6. Second held-out test (V021–V050, written after the prompts were fixed): repeat steps 3–5 with `--set extra30`, or `--set all50` for everything together.
7. Independent labels: `python kappa_agreement.py --make-sheet`, give `rater_sheet.csv` to an athletic trainer / nurse / physician, then `python kappa_agreement.py --rater rater_sheet.csv`.
8. Booth backup: `py build_demo_into_html.py web/classifier_flask.html --responses saved_responses_all50.json`, then tick **Demo mode** in the page.

Every run saves `results/run_metadata_<set>.json` (exact model that answered, date/time, prompt version, temperature) and `results/saved_responses_<set>.json` — use these for the SSEF AI-use documentation.

## Files
| File | What it is |
|---|---|
| `rule.py` | The decision rule, temperature rounding, caveats, and scoring policy — one place |
| `prompts.py` | The two yes/no questions (`v1_original`, `v2_expanded`) |
| `full_classifier_groq.py` | AI-assisted classifier + evaluation runner |
| `baseline_classifier.py` | Temperature-only baseline |
| `keyword_classifier.py` | Keyword-search baseline (naive + negation-aware) |
| `compare_classifiers.py` | McNemar, Heat Stroke recall, error analysis |
| `kappa_agreement.py` | Independent-labeler agreement |
| `stats_utils.py` | Wilson interval, confusion matrix, McNemar, kappa, consistency |
| `data/vignettes_test20.json` | **Locked** 20 test cases |
| `data/vignettes_extra30.json` | 30 more held-out cases (V021–V050) |
| `data/dev_cases.json` | 10 practice cases (V101–V110) for prompt experiments only |
| `check_parity.py` | Confirms the web page and Python give identical answers (fake AI replies) |
| `tests/` | Rule/statistics tests and a fake-server test of the runner |

## Scoring policy (fixed before running)
* Primary metric: strict exact match. "Uncertain — flag for review" is its own class.
* Dangerous miss: a true Heat Stroke called Heat Exhaustion, Not Heat Illness, or unreadable. A flag for review is not a dangerous miss.
* An AI reply that can't be read after 2 retries is counted wrong and reported.
* Official unit is Celsius. Temperatures are rounded to 0.1 °C (half-up) before comparing; the threshold is *strictly above* 40.5 °C, so exactly 40.5 is not above, and 105.0 °F (40.56 °C → 40.6) is.
