"""Copies the vignettes from the JSON files into the web page's vignette cards,
so the page and the Python code always use the exact same text. (Demo mode
needs the text to match exactly, word for word.)

    py build_vignettes_into_html.py web/classifier_flask.html            (all 50)
    py build_vignettes_into_html.py web/classifier_flask.html --set test20

Run it again any time you edit vignettes.json or data/vignettes_extra30.json.
"""
import json
import re
import sys

from vignette_sets import load_set

args = sys.argv[1:]
set_name = "all50"
if "--set" in args:
    i = args.index("--set")
    set_name = args[i + 1]
    del args[i:i + 2]
if not args:
    raise SystemExit("usage: py build_vignettes_into_html.py web/classifier_flask.html [--set all50|test20|extra30]")

vignettes = load_set(set_name)
lines = []
for v in vignettes:
    card = {key: v[key] for key in ("id", "symptom_text", "temperature_c", "temperature_source", "true_label")}
    lines.append("  " + json.dumps(card, ensure_ascii=False) + ",")
js_array = "const VIGNETTES = [\n" + "\n".join(lines) + "\n];"

for path in args:
    src = open(path, encoding="utf-8").read()
    new, count = re.subn(r"const VIGNETTES = \[.*?\n\];", lambda _: js_array, src, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{path}: could not find the VIGNETTES list")
    open(path, "w", encoding="utf-8").write(new)
    print(f"{path}: now has {len(vignettes)} vignette cards ({set_name})")
