"""Checks that classifier.html (JavaScript) and the Python code give EXACTLY
the same answers (review item 14).

IN PLAIN WORDS: the older web pages (classifier_1.html, classifier.html) ran
the whole classifier in JavaScript, a copy of the Python code. Copies can
drift apart, so this script opens the page in a hidden browser and checks
both give identical answers. The newer Flask page (classifier_flask.html)
asks the Python code directly, so it can't drift; only its demo mode still
uses the JavaScript copy.

IMPORTANT: this uses FAKE, deterministic AI replies so it can run anywhere
without a key or internet. It therefore tests everything around the AI -- the
prompt text, the decision rule, the temperature rounding, the caveats, the
demo-mode replay -- not the AI itself. (Real AI behavior is measured by
full_classifier_groq.py.)

Needs:  pip install playwright && playwright install chromium
Run:    python check_parity.py legacy/classifier_for_sharing.html
        (set PW_CHROMIUM=/path/to/chromium if Playwright can't find a browser)
"""
import json, os, sys, tempfile, shutil, subprocess, asyncio
from playwright.sync_api import sync_playwright

import prompts, rule
import full_classifier_groq as fgq
from vignette_sets import load_set

html_files = sys.argv[1:] or ["legacy/classifier_for_sharing.html"]
here = os.path.dirname(os.path.abspath(__file__))
failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("  FAIL:", msg)


def fake_reply(flag):
    return f"Answer: {flag}\nReason: fake reply for parity test."


vignettes = load_set("all50")
# Deliberately flip some flags so every label (incl. Uncertain) shows up on both sides.
flags = {}
for k, v in enumerate(vignettes):
    cns, heat = v["true_cns"], v["true_heat_symptoms"]
    if k % 7 == 3:
        cns = 1 - cns
    if k % 5 == 2:
        heat = 1 - heat
    flags[v["id"]] = (cns, heat)

reply_for_prompt = {}
for v in vignettes:
    cns, heat = flags[v["id"]]
    reply_for_prompt[prompts.build_prompt("cns", v["symptom_text"])] = fake_reply(cns)
    reply_for_prompt[prompts.build_prompt("heat", v["symptom_text"])] = fake_reply(heat)
CONTEXT = "Sitting in a classroom all morning."
CTX_TEXT = "A student is confused about where she is."
reply_for_prompt[prompts.build_prompt("cns", CTX_TEXT, CONTEXT)] = fake_reply(1)
reply_for_prompt[prompts.build_prompt("heat", CTX_TEXT, CONTEXT)] = fake_reply(0)
reply_for_prompt[prompts.build_context_prompt(CONTEXT)] = fake_reply(0)

# ---- Python side, using the same fake replies ----
fgq.call_llm = lambda prompt, *a, **k: (reply_for_prompt[prompt], "fake-model")
py_labels = {}
for v in vignettes:
    r = fgq.classify_detailed(v["symptom_text"], v["temperature_c"], v.get("temperature_source", ""))
    py_labels[v["id"]] = (r["label"], r["caveats"])
py_ctx = fgq.classify_detailed(CTX_TEXT, 38.0, "oral", CONTEXT)
print("Python labels seen:", sorted({l for l, _ in py_labels.values()}))

# ---- baked-in demo copies ----
tmp = tempfile.mkdtemp()
saved = [{"prompt": p, "reply": r, "model": "fake-model"} for p, r in reply_for_prompt.items()]
json.dump(saved, open(os.path.join(tmp, "saved.json"), "w"))

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=os.environ.get("PW_CHROMIUM"))
    for html in html_files:
        print(f"\n=== {html} ===")
        html_path = os.path.abspath(html)
        is_sharing_copy = "PASTE_YOUR_GROQ_API_KEY_HERE" in open(html_path, encoding="utf-8").read().split("const GROQ_API_KEY =")[1][:60]

        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        def handle(route):
            body = route.request.post_data_json
            check(body.get("temperature") == 0, "request did not send temperature: 0")
            prompt = body["messages"][0]["content"]
            if prompt not in reply_for_prompt:
                route.fulfill(status=500, body="no fake reply for this prompt")
                return
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"model": "fake-model", "choices": [{"message": {"content": reply_for_prompt[prompt]}}]}))
        page.route("https://api.groq.com/**", handle)
        page.goto("file://" + html_path)

        # key box shows only when the key is the placeholder (item 13)
        key_box_visible = page.is_visible("#keyField")
        check(key_box_visible == is_sharing_copy, f"key box visibility wrong (sharing copy={is_sharing_copy}, visible={key_box_visible})")
        if is_sharing_copy:
            page.fill("#apiKeyInput", "gsk_fake_typed_key")

        # 1. prompt text identical
        for kind, qvar in (("cns", "cnsQuestion"), ("heat", "heatQuestion")):
            for text, ctx in (("A runner is confused.", ""), ("Odd & <tag> text", "ran sprints in pads")):
                js = page.evaluate("([q,t,c]) => buildPrompt(window[q] || eval(q), t, c)", [qvar, text, ctx])
                check(js == prompts.build_prompt(kind, text, ctx), f"{kind} prompt text differs (ctx={ctx!r})")
        check(page.evaluate("() => buildContextPrompt('Ran sprints.')") == prompts.build_context_prompt("Ran sprints."), "context prompt differs")

        # 2. rule + rounding across a fine temperature grid, incl. Fahrenheit conversions
        temps = [round(36 + i * 0.05, 2) for i in range(0, 150)] + [rule.fahrenheit_to_celsius(104 + i * 0.1) for i in range(0, 25)]
        js_norm = page.evaluate("(ts) => ts.map(t => normalizeTempC(t))", temps)
        for t, j in zip(temps, js_norm):
            check(abs(float(rule.normalize_temp_c(t)) - j) < 1e-9, f"normalize differs at {t}: py={rule.normalize_temp_c(t)} js={j}")
        combos = [(c, h, t) for c in (0, 1) for h in (0, 1) for t in (36.0, 40.4, 40.5, 40.51, 40.55, 40.6, rule.fahrenheit_to_celsius(105.0), 43.0)]
        js_rule = page.evaluate("(cs) => cs.map(([c,h,t]) => applyRule(c,h,t))", combos)
        for (c, h, t), j in zip(combos, js_rule):
            check(rule.apply_rule(c, h, t) == j, f"rule differs for cns={c} heat={h} t={t}")

        # 3. caveat text identical
        cav_cases = [(lab, src, cns, ctx, exp) for lab in rule.ALL_LABELS for src in ("", "oral", "rectal", "forehead") for cns in (0, 1)
                     for ctx, exp in (("", None), ("ran sprints", 1), ("sat in class", 0))]
        js_cav = page.evaluate("(cs) => cs.map(([l,s,c,x,e]) => buildCaveats(l,s,c,x,e))", cav_cases)
        for case, j in zip(cav_cases, js_cav):
            check(rule.build_caveats(*case) == j, f"caveats differ for {case}")

        # 4. all 50 vignettes end to end (live-API code path with fake replies)
        js_all = page.evaluate("""async (vs) => {
            const out = {};
            for (const v of vs) { const r = await classifyHeatIllness(v.symptom_text, v.temperature_c, "", v.temperature_source); out[v.id] = [r.label, r.caveats]; }
            return out; }""", [{k: v[k] for k in ("id", "symptom_text", "temperature_c", "temperature_source")} for v in vignettes])
        mism = [i for i in py_labels if js_all[i][0] != py_labels[i][0] or js_all[i][1] != py_labels[i][1]]
        check(not mism, f"JS and Python disagree on: {mism}")
        print(f"  {len(vignettes) - len(mism)}/{len(vignettes)} vignettes identical (label + caveats) between JavaScript and Python")

        # 5. context question path
        js_ctx = page.evaluate("async ([t,c]) => { const r = await classifyHeatIllness(t, 38.0, c, 'oral'); return [r.label, r.caveats]; }", [CTX_TEXT, CONTEXT])
        check(js_ctx[0] == py_ctx["label"] and js_ctx[1] == py_ctx["caveats"], "context-question path differs")
        check(any("concussion" in c for c in js_ctx[1]), "context caveat (concussion) missing in JS")

        # 6. demo mode: no network at all, replies baked into a copy of the page
        demo_copy = os.path.join(tmp, os.path.basename(html))
        shutil.copy(html_path, demo_copy)
        subprocess.run([sys.executable, os.path.join(here, "build_demo_into_html.py"), demo_copy, "--responses", os.path.join(tmp, "saved.json")], check=True, capture_output=True)
        dpage = browser.new_page()
        derrors = []
        dpage.on("pageerror", lambda e: derrors.append(str(e)))
        calls = []
        dpage.route("**/*", lambda route: (calls.append(route.request.url), route.abort()) if route.request.url.startswith("http") else route.continue_())
        dpage.goto("file://" + demo_copy)
        dpage.check("#demoMode")
        d_all = dpage.evaluate("""async (vs) => {
            const out = {};
            for (const v of vs) { const r = await classifyHeatIllness(v.symptom_text, v.temperature_c, "", v.temperature_source); out[v.id] = r.label; }
            return out; }""", [{k: v[k] for k in ("id", "symptom_text", "temperature_c", "temperature_source")} for v in vignettes])
        check(all(d_all[i] == py_labels[i][0] for i in py_labels), "demo-mode labels differ from Python")
        check(not [u for u in calls if "groq" in u], "demo mode tried to reach the network")
        # a missing replay gives a clear error rather than silently calling the API
        msg = dpage.evaluate("async () => { try { await classifyHeatIllness('never recorded text', 38, '', ''); return 'no error'; } catch (e) { return e.message; } }")
        check("no saved response" in msg, f"missing-replay message unexpected: {msg}")
        check(not derrors, f"page errors in demo copy: {derrors}")
        print(f"  demo mode: {len(vignettes)}/{len(vignettes)} replayed offline; clear error for unrecorded cases")
        dpage.close()

        check(not errors, f"page errors: {errors}")
        page.close()
    browser.close()

shutil.rmtree(tmp, ignore_errors=True)
print("\nPARITY CHECK:", "PASSED" if not failures else f"FAILED ({len(failures)} problem(s))")
sys.exit(1 if failures else 0)
