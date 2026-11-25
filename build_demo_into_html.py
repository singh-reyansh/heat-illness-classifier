"""Bakes the AI's saved replies into classifier.html so 'Demo mode' works
with no internet and no API key (review item 14: a plan for no Wi-Fi at the
booth).

IN PLAIN WORDS: full_classifier_groq.py saves every question it asked the AI
and the AI's exact reply. This script writes that list INTO the web page
(between the /*DEMO_START*/ and /*DEMO_END*/ markers) as a lookup table:
question -> reply. In demo mode the page looks up the reply instead of asking
the internet, which is why it only works for questions that were recorded.

    1. python full_classifier_groq.py --set all50           (needs your key + internet, once)
    2. python build_demo_into_html.py web/classifier_flask.html --responses saved_responses_all50.json
    3. In the page, tick "Demo mode" and click any of the vignette cards.

--responses can be given more than once to combine runs, e.g.
    --responses saved_responses_test20.json --responses saved_responses_extra30.json

Only REAL replies saved by step 1 are used -- nothing is invented. Because
the replies are looked up by the full prompt text, a demo reply can only be
shown for the exact question it was recorded for.
"""
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

if len(sys.argv) < 2:
    raise SystemExit("usage: python build_demo_into_html.py web/classifier_flask.html [more.html ...] [--responses saved_responses_test20.json]")

args = sys.argv[1:]
responses_files = []
while "--responses" in args:
    i = args.index("--responses")
    responses_files.append(args[i + 1])
    del args[i:i + 2]
if not responses_files:
    responses_files = ["saved_responses_test20.json"]

table = {}
for responses_file in responses_files:
    if not os.path.exists(responses_file):
        responses_file = os.path.join(RESULTS_DIR, os.path.basename(responses_file))
    with open(responses_file, encoding="utf-8") as f:
        calls = json.load(f)
    for c in calls:
        table[c["prompt"]] = {"reply": c["reply"], "model": c.get("model", "")}

for path in args:
    src = open(path, encoding="utf-8").read()
    a, b = src.index("/*DEMO_START*/"), src.index("/*DEMO_END*/")
    new = src[:a] + "/*DEMO_START*/" + json.dumps(table, ensure_ascii=False) + src[b:]
    open(path, "w", encoding="utf-8").write(new)
    print(f"{path}: baked in {len(table)} saved replies")
