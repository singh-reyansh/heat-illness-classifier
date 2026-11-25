"""Plain asserts -- run with:  python tests/test_all.py   (from the project folder)

IN PLAIN WORDS: automatic checks. Each `assert` line says "this must be
true"; if any is false, the script stops with an error showing which one.
They test the tricky edge cases (exactly 40.5 C, 105 F, rounding, the
statistics formulas) so a change that breaks something gets caught right away.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rule, prompts, stats_utils as s
from rule import *
from vignette_sets import load_set
from keyword_classifier import classify_keyword

# --- rule edge cases (review item 12) ---
assert apply_rule(0, 1, 40.5) == LABEL_HEAT_EXHAUSTION          # exactly 40.5 is NOT above
assert apply_rule(0, 1, 40.6) == LABEL_UNCERTAIN
assert apply_rule(0, 1, 40.51) == LABEL_HEAT_EXHAUSTION         # rounds to 40.5 (0.1 C resolution)
assert apply_rule(0, 0, 41.0) == LABEL_UNCERTAIN
assert apply_rule(1, 0, 36.0) == LABEL_HEAT_STROKE              # CNS alone wins
assert apply_rule(0, 0, 38.0) == LABEL_NOT_HEAT_ILLNESS
assert is_above_threshold(fahrenheit_to_celsius(105.0))          # 105.0 F = 40.56 C -> counted above
assert not is_above_threshold(fahrenheit_to_celsius(104.9))      # 40.50 C
assert str(normalize_temp_c(40.55)) == "40.6" and str(normalize_temp_c(40.45)) == "40.5"

# --- all stored labels agree with the rule given their own flags ---
for name in ("test20", "extra30", "all50", "dev"):
    for v in load_set(name):
        assert apply_rule(v["true_cns"], v["true_heat_symptoms"], v["temperature_c"]) == v["true_label"], v["id"]
assert len(load_set("test20")) == 20 and len(load_set("extra30")) == 30 and len(load_set("all50")) == 50
ids = [v["id"] for v in load_set("all50")]
assert len(set(ids)) == 50
dev_ids = {v["id"] for v in load_set("dev")}
assert not dev_ids & set(ids), "dev cases must not overlap test cases"

# --- scoring policy (item 7 / 8) ---
assert not is_dangerous_miss(LABEL_HEAT_STROKE, LABEL_UNCERTAIN)
assert is_dangerous_miss(LABEL_HEAT_STROKE, LABEL_HEAT_EXHAUSTION)
assert is_dangerous_miss(LABEL_HEAT_STROKE, LABEL_UNPARSEABLE)
assert not is_exact_match(LABEL_HEAT_STROKE, LABEL_UNCERTAIN)
hs = s.heat_stroke_summary([LABEL_HEAT_STROKE] * 4, [LABEL_HEAT_STROKE, LABEL_UNCERTAIN, LABEL_HEAT_EXHAUSTION, LABEL_UNPARSEABLE])
assert (hs["strict_hits"], hs["flagged_for_review"], hs["dangerous_misses"]) == (1, 1, 2)

# --- statistics ---
assert abs(s.mcnemar_exact([True] * 8, [False] * 8)["p_value"] - 2 / 256) < 1e-12
assert s.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
assert abs(s.cohens_kappa(list("aabbab"), list("aabbba")) - 1 / 3) < 1e-9
assert s.cohens_kappa(list("ab"), list("ab")) == 1.0
c = s.consistency_report([["a", "b", "c"], ["a", "c", "c"], ["a", "b", "c"]])
assert c["majority_labels"] == ["a", "b", "c"] and c["identical_cases"] == 2
assert s.wilson_confidence_interval(19, 20)[0] > 0.76 and abs(s.wilson_confidence_interval(8, 20)[0] - 0.219) < 0.001

# --- prompts ---
p = prompts.build_prompt("cns", "A runner is confused.")
assert p.endswith("Reason: one sentence") and "Description: A runner is confused.\n\nRespond" in p
assert "collapse" in p and "seizure" in p and "irritable" in p and "delirium" in p      # item 16
assert "Additional context" in prompts.build_prompt("heat", "x", "ran sprints")
assert prompts.build_prompt("cns", "x", version="v1_original") != prompts.build_prompt("cns", "x")

# --- caveats (items 3, 4) ---
assert build_caveats(LABEL_HEAT_EXHAUSTION, "oral") and "cannot rule out heat stroke" in build_caveats(LABEL_HEAT_EXHAUSTION, "oral")[0]
assert build_caveats(LABEL_HEAT_EXHAUSTION, "rectal") == []
assert "not specified" in build_caveats(LABEL_NOT_HEAT_ILLNESS, "")[0]
assert any("CNS signs alone" in c for c in build_caveats(LABEL_HEAT_STROKE, "rectal", cns_flag=1))
assert any("concussion" in c for c in build_caveats(LABEL_HEAT_STROKE, "rectal", 1, "sat in a classroom", 0))

# --- keyword baseline basics ---
assert classify_keyword("He is confused and sweating", 38.0)[0] == LABEL_HEAT_STROKE
assert classify_keyword("She is not confused, just sweating", 38.0, negation_aware=True)[0] == LABEL_HEAT_EXHAUSTION
assert classify_keyword("She is not confused, just sweating", 38.0, negation_aware=False)[0] == LABEL_HEAT_STROKE
assert classify_keyword("Fine, chatting", 38.0)[0] == LABEL_NOT_HEAT_ILLNESS

# --- the word shown on the web page is the one that made the flag 1 ---
from keyword_classifier import matched_word, CNS_KEYWORDS, HEAT_KEYWORDS
assert matched_word("Acting irritable today", CNS_KEYWORDS, False) == "irritable"
assert matched_word("She is not confused, just sweating", CNS_KEYWORDS, True) is None
for v in load_set("all50"):
    for neg in (False, True):
        _, cns, heat = classify_keyword(v["symptom_text"], v["temperature_c"], neg)
        assert (matched_word(v["symptom_text"], CNS_KEYWORDS, neg) is not None) == (cns == 1), v["id"]
        assert (matched_word(v["symptom_text"], HEAT_KEYWORDS, neg) is not None) == (heat == 1), v["id"]

# --- web server: the no-AI baselines route (no API key needed) ---
import server
client = server.app.test_client()
r = client.post("/api/baselines", json={"symptom_text": "He is confused and sweating", "temperature_c": 38.0})
assert r.status_code == 200
assert r.get_json()["keyword_naive"]["label"] == LABEL_HEAT_STROKE and r.get_json()["keyword_naive"]["cnsWord"] == "confused"
assert r.get_json()["temperature_only"]["label"] == LABEL_HEAT_EXHAUSTION
assert client.post("/api/baselines", json={"symptom_text": "", "temperature_c": 38.0}).status_code == 400
assert client.post("/api/baselines", json={"symptom_text": "tired", "temperature_c": "hot"}).status_code == 400

# --- results baked into the web page match the results CSVs ---
import build_results_into_html
res = build_results_into_html.build("all50")
assert res["methods"][0]["key"] == "ai" and res["methods"][0]["correct"] == sum(c["ai"]["label"] == c["true"] for c in res["cases"])
assert all(m["n"] == len(res["cases"]) for m in res["methods"])
assert all(sum(map(sum, m["matrix"])) == m["n"] for m in res["methods"])

print("ALL TESTS PASSED")
