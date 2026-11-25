"""The AI-assisted classifier (Groq API) plus the evaluation runner.

IN PLAIN WORDS -- this is the project's main engine. For each case it:
  1. builds the CNS question and the heat question (prompts.py)
  2. sends each one to the Groq LLM over the internet (call_llm)
  3. reads the "Answer: 1 or 0" and "Reason: ..." out of the reply
     (parse_response / parse_reason), asking again up to 2 times if the
     reply is in the wrong format (ask_binary_question)
  4. lets the fixed rule pick the label (rule.apply_rule)
classify_detailed() does steps 1-4 for one case. run_evaluation() does it
for a whole set of vignettes, prints the score, and saves the results CSV,
the run details, and every AI reply (used later by demo mode).
The web app (server.py) calls the SAME classify_detailed(), so the app is
exactly what was tested.

What changed after expert review (item numbers refer to the review list):
  10  temperature=0 (as repeatable as the API allows) and --runs N to repeat
      every case and report how consistent the answers were. If a reply still
      can't be read after the retries, the case is counted WRONG
      ("Unparseable") and the number of such cases is reported.
  11  The exact model ID that answered (taken from the API's own reply), the
      prompt version, the date/time and the settings are saved to
      run_metadata_<set>.json for the SSEF AI-use documentation. Set
      GROQ_MODEL to pin one model; otherwise the first of MODEL_CANDIDATES
      that works is used, and which one it was is recorded.
  14  Every raw AI reply is saved to saved_responses_<set>.json. Run
      build_demo_into_html.py afterwards to bake them into classifier.html so
      "Demo mode" works with no internet.
  16  Prompts live in prompts.py (v1_original / v2_expanded).
  3/4 Results also carry plain-language caveats from rule.build_caveats().

Usage (from this folder):
    set GROQ_API_KEY=...            (Windows)   /   export GROQ_API_KEY=...  (Mac/Linux)
    python full_classifier_groq.py --set test20 --runs 3
"""
import argparse
import csv
import datetime
import json
import os
import re
import time

import requests

import prompts
import rule
from vignette_sets import load_set, report_classifier, RESULTS_DIR
from stats_utils import consistency_report

API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Groq retires models and its docs lag behind what a key can actually use
# (groq/compound-mini was shut down; llama-3.3-70b-versatile returned
# "model_not_found" on this account even though the docs listed it). So: try
# these in order, remember the first that works, and RECORD which one it was.
# Set GROQ_MODEL to force exactly one model instead.
MODEL_CANDIDATES = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TEMPERATURE = 0

_active_model_index = 0
CALL_LOG = []  # every raw call: prompt, reply, model that answered -- saved for replay/demo


def _api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Set it in your terminal before running this script.")
    return key


def _candidates():
    pinned = os.environ.get("GROQ_MODEL")
    return [pinned] if pinned else MODEL_CANDIDATES


def call_llm(prompt, max_rate_limit_retries=6):
    """One API call -> (reply_text, model_that_answered). Waits and retries on
    HTTP 429 (rate limit); moves to the next candidate model on 404
    'model_not_found'."""
    # IN PLAIN WORDS: send the prompt to Groq over the internet and get the
    # AI's reply back. Two things can go wrong, and both are handled:
    #   * HTTP 429 "too many requests": Groq limits how fast we can ask, so
    #     we wait and try again.
    #   * HTTP 404 "model not found": Groq retired that model, so we move on
    #     to the next model in MODEL_CANDIDATES.
    global _active_model_index
    models = _candidates()
    i = min(_active_model_index, len(models) - 1)
    while i < len(models):
        for attempt in range(max_rate_limit_retries + 1):
            # The request: which model, temperature 0 (least random), and the
            # prompt as a "user" message. The API key proves it is our account.
            response = requests.post(
                url=API_URL,
                headers={"Authorization": f"Bearer {_api_key()}"},
                json={"model": models[i], "temperature": TEMPERATURE, "messages": [{"role": "user", "content": prompt}]},
                timeout=90,
            )
            if response.status_code == 429 and attempt < max_rate_limit_retries:
                retry_after = response.headers.get("Retry-After")
                wait = int(float(retry_after)) if retry_after else min(10 * (2 ** attempt), 60)
                print(f"      (rate limited — waiting {wait}s, attempt {attempt + 1}/{max_rate_limit_retries})")
                time.sleep(wait)
                continue
            break
        if response.status_code == 404 and "model_not_found" in response.text and i < len(models) - 1:
            print(f"      (model {models[i]!r} not available for this key — trying {models[i + 1]!r})")
            i += 1
            _active_model_index = i
            continue
        response.raise_for_status()
        data = response.json()
        _active_model_index = i
        time.sleep(1)  # stay under the per-minute limit
        text = data["choices"][0]["message"]["content"]
        answered_by = data.get("model", models[i])  # the API's own statement of which model replied
        CALL_LOG.append({"prompt": prompt, "reply": text, "model": answered_by, "requested_model": models[i],
                         "system_fingerprint": data.get("system_fingerprint")})
        return text, answered_by
    raise RuntimeError("No working Groq model found for this API key.")


def parse_response(response_text):
    # Finds "Answer: 1" or "Answer: 0" in the reply. In the pattern, \s* means
    # "any spaces" and ([01]) captures the single digit 0 or 1.
    match = re.search(r"Answer:\s*([01])", response_text)
    if not match:
        raise ValueError(f"Could not parse LLM response:\n{response_text}")
    return int(match.group(1))


def parse_reason(response_text):
    match = re.search(r"Reason:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()


def ask_binary_question(prompt, max_retries=2):
    """Returns (flag, reason). Raises ValueError if the reply can't be read
    after max_retries + 1 attempts (callers decide what to do about that)."""
    last = None
    for attempt in range(max_retries + 1):
        text, _model = call_llm(prompt)
        last = text
        try:
            return parse_response(text), parse_reason(text)
        except ValueError:
            if attempt < max_retries:
                print(f"      (retrying — unparseable response: {text[:60]!r})")
    raise ValueError(f"Could not parse LLM response after {max_retries + 1} attempts:\n{last}")


def classify_detailed(symptom_text, temperature_c, temp_source="", context_text="", prompt_version=prompts.DEFAULT_PROMPT_VERSION):
    """Full result as a dict. Never raises on an unreadable AI reply: the
    label becomes 'Unparseable — no answer' and unparseable=True."""
    result = {"unparseable": False, "cns_flag": None, "heat_flag": None, "cns_reason": "", "heat_reason": "", "caveats": []}
    # Step 1 + 2: ask the AI the two yes/no questions.
    try:
        cns_flag, cns_reason = ask_binary_question(prompts.build_prompt("cns", symptom_text, context_text, prompt_version))
        heat_flag, heat_reason = ask_binary_question(prompts.build_prompt("heat", symptom_text, context_text, prompt_version))
    except ValueError as err:
        result.update(unparseable=True, label=rule.LABEL_UNPARSEABLE, cns_reason=str(err)[:200])
        return result

    # Optional third question: only when a CNS sign was found AND the user said
    # what the person was doing. It never changes the label; it only adds a
    # warning if the note doesn't mention heat or exercise (other causes of
    # confusion, like a concussion, can look similar).
    exposure_flag = None
    if cns_flag == 1 and (context_text or "").strip():
        try:
            exposure_flag, _ = ask_binary_question(prompts.build_context_prompt(context_text))
        except ValueError:
            exposure_flag = None

    # Step 3: the fixed rule (not the AI) picks the label.
    label = rule.apply_rule(cns_flag, heat_flag, temperature_c)
    result.update(label=label, cns_flag=cns_flag, heat_flag=heat_flag, cns_reason=cns_reason, heat_reason=heat_reason,
                  caveats=rule.build_caveats(label, temp_source, cns_flag, context_text, exposure_flag))
    return result


def classify_heat_illness(symptom_text, temperature_c):
    """Original 5-tuple interface (used by app.py and demo.py)."""
    r = classify_detailed(symptom_text, temperature_c)
    if r["unparseable"]:
        raise ValueError(r["cns_reason"])
    return r["label"], r["cns_flag"], r["heat_flag"], r["cns_reason"], r["heat_reason"]


def run_evaluation(set_name, runs, prompt_version, save_prefix="full_classifier_groq"):
    vignettes = load_set(set_name)
    started = datetime.datetime.now(datetime.timezone.utc)
    all_runs = []  # list of runs; each a list of result dicts
    for run_number in range(1, runs + 1):
        print(f"\n--- Run {run_number} of {runs} ({set_name}, prompts {prompt_version}) ---")
        one_run = []
        for v in vignettes:
            r = classify_detailed(v["symptom_text"], v["temperature_c"], v.get("temperature_source", ""), "", prompt_version)
            one_run.append(r)
            print(f"{v['id']:<6}{r['label']:<48}(true: {v['true_label']})")
        all_runs.append(one_run)

    # With --runs 3, every case was classified 3 times. Check how often the
    # label stayed the same, and use the majority vote as the final label.
    run_labels = [[r["label"] for r in one] for one in all_runs]
    cons = consistency_report(run_labels)
    final_labels = cons["majority_labels"]  # majority vote; with runs=1 it is just that run

    rows = []
    for i, v in enumerate(vignettes):
        first = all_runs[0][i]
        rows.append({
            "id": v["id"], "symptom_text": v["symptom_text"], "temperature_c": v["temperature_c"],
            "temperature_source": v.get("temperature_source", ""),
            "predicted_label": final_labels[i], "true_label": v["true_label"],
            "match": "correct" if final_labels[i] == v["true_label"] else "WRONG",
            "labels_each_run": " | ".join(rl[i] for rl in run_labels),
            "cns_flag": first["cns_flag"], "cns_reason": first["cns_reason"],
            "heat_symptoms_flag": first["heat_flag"], "heat_symptoms_reason": first["heat_reason"],
            "unparseable_in_any_run": any(one[i]["unparseable"] for one in all_runs),
            "nonrectal_caveat": any("non-rectal" in c or "not specified" in c for c in first["caveats"]),
        })

    title = f"AI_assisted_{prompt_version}" + (f"_majority_of_{runs}" if runs > 1 else "")
    report_classifier(title, rows, f"{save_prefix}_results_{set_name}.csv")

    n_unparse_cases = sum(r["unparseable_in_any_run"] for r in rows)
    n_unparse_calls = sum(r["unparseable"] for one in all_runs for r in one)
    print(f"\nUnparseable replies (after 2 retries): {n_unparse_calls} case-run(s) across {n_unparse_cases} case(s). Counted as WRONG.")
    if runs > 1:
        per_run_acc = [sum(l == v['true_label'] for l, v in zip(rl, vignettes)) / len(vignettes) * 100 for rl in run_labels]
        print(f"Per-run accuracy: " + ", ".join(f"{a:.1f}%" for a in per_run_acc))
        print(f"Consistency: {cons['identical_cases']}/{cons['n_cases']} cases got the identical label in all {runs} runs "
              f"({cons['identical_fraction'] * 100:.1f}%).")
    flagged = sum(r["nonrectal_caveat"] for r in rows)
    print(f"Cases with a non-rectal / unspecified temperature caveat: {flagged}/{len(rows)}")

    # Save a record of HOW this run was done (model, date, settings) for the
    # science fair's AI-use documentation, plus every raw AI reply so the
    # results can be checked later and replayed in demo mode.
    models_used = sorted({c["model"] for c in CALL_LOG})
    meta = {
        "run_started_utc": started.isoformat(), "run_finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "set": set_name, "n_cases": len(vignettes), "runs": runs, "prompt_version": prompt_version,
        "temperature_setting": TEMPERATURE, "models_that_answered": models_used,
        "requested_models_tried": _candidates(), "pinned_via_GROQ_MODEL": os.environ.get("GROQ_MODEL"),
        "system_fingerprints_seen": sorted({str(c["system_fingerprint"]) for c in CALL_LOG}),
        "total_api_calls": len(CALL_LOG), "unparseable_case_runs": n_unparse_calls,
        "note": "Model IDs come from the API's own reply. Plain (non-'compound') models do not call tools such as web search.",
    }
    with open(os.path.join(RESULTS_DIR, f"run_metadata_{set_name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"saved_responses_{set_name}.json"), "w", encoding="utf-8") as f:
        json.dump(CALL_LOG, f, indent=1)
    print(f"Saved results/run_metadata_{set_name}.json (model, date, settings) and results/saved_responses_{set_name}.json (for demo mode).")
    print(f"Model(s) that answered: {', '.join(models_used)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="test20", help="test20 (default), extra30, all50, or dev")
    parser.add_argument("--runs", type=int, default=1, help="repeat every case this many times (3-5 recommended for the consistency check)")
    parser.add_argument("--prompts", default=prompts.DEFAULT_PROMPT_VERSION, choices=list(prompts.PROMPT_VERSIONS))
    args = parser.parse_args()
    run_evaluation(args.set, args.runs, args.prompts)


if __name__ == "__main__":
    main()
