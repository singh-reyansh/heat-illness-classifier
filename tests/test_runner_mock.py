"""End-to-end test of full_classifier_groq.py + compare_classifiers.py with a
FAKE Groq server (no key, no internet). Checks the plumbing: model fallback
on 404, temperature=0 in every request, repeated runs + consistency,
unparseable replies counted as WRONG, metadata/replay files written, and the
comparison script. It does NOT test how good the real AI is.

Run from the project folder:  python tests/test_runner_mock.py
"""
import json, os, re, shutil, subprocess, sys, tempfile

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
work = tempfile.mkdtemp()
shutil.copytree(SRC, os.path.join(work, "p"), ignore=shutil.ignore_patterns("*.csv", "run_metadata*", "saved_responses*", "__pycache__", "*.html"))
os.chdir(os.path.join(work, "p"))
sys.path.insert(0, os.getcwd())
os.environ["GROQ_API_KEY"] = "gsk_fake"

import requests
import full_classifier_groq as fgq
from vignette_sets import load_set

truth = {v["symptom_text"]: v for v in load_set("test20")}
requests_seen = []
call_counter = {"n": 0}


class FakeResp:
    def __init__(self, status, payload, text=None, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}
        self.text = text if text is not None else json.dumps(payload)
    def json(self): return self._p
    def raise_for_status(self):
        if self.status_code >= 400: raise requests.HTTPError(f"{self.status_code}")


def fake_post(url, headers=None, json=None, timeout=None):
    requests_seen.append(json)
    call_counter["n"] += 1
    if json["model"] == "openai/gpt-oss-20b":
        return FakeResp(404, {}, text='{"error":{"code":"model_not_found"}}')
    prompt = json["messages"][0]["content"]
    desc = re.search(r"Description: (.*)\n", prompt).group(1)
    v = truth[desc]
    flag = v["true_cns"] if "central nervous system" in prompt else v["true_heat_symptoms"]
    if v["id"] == "V004":            # this case always gets an unreadable reply
        reply = "I cannot answer that."
    elif v["id"] == "V009" and call_counter["n"] % 2 == 0:   # flaky case: flips on alternate calls
        reply = f"Answer: {1 - flag}\nReason: flaky."
    else:
        reply = f"Answer: {flag}\nReason: fake reason."
    return FakeResp(200, {"model": json["model"], "system_fingerprint": "fp_fake", "choices": [{"message": {"content": reply}}]})

requests.post = fake_post
fgq.time.sleep = lambda s: None

fgq.run_evaluation("test20", runs=2, prompt_version="v2_expanded")

assert all(r["temperature"] == 0 for r in requests_seen), "every request must send temperature=0"
meta = json.load(open("results/run_metadata_test20.json"))
assert meta["models_that_answered"] == ["openai/gpt-oss-120b"], meta
assert meta["temperature_setting"] == 0 and meta["runs"] == 2 and meta["prompt_version"] == "v2_expanded"
assert meta["unparseable_case_runs"] == 2, meta                      # V004, both runs
saved = json.load(open("results/saved_responses_test20.json"))
assert saved and all(c["model"] == "openai/gpt-oss-120b" for c in saved)
import csv
rows = {r["id"]: r for r in csv.DictReader(open("results/full_classifier_groq_results_test20.csv", encoding="utf-8"))}
assert rows["V004"]["predicted_label"].startswith("Unparseable") and rows["V004"]["match"] == "WRONG"
assert rows["V001"]["match"] == "correct"

subprocess.run([sys.executable, "baseline_classifier.py"], check=True, capture_output=True)
subprocess.run([sys.executable, "keyword_classifier.py"], check=True, capture_output=True)
out = subprocess.run([sys.executable, "compare_classifiers.py", "--set", "test20"], check=True, capture_output=True, text=True).stdout
assert "McNemar" in out and "ERROR ANALYSIS" in out and "V004" in out and "p =" in out
print(out[:2600])
shutil.rmtree(work, ignore_errors=True)
print("\nRUNNER + COMPARE MOCK TEST PASSED")
