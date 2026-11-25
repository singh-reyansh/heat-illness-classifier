"""Engineering design log for the heat illness classifier (SSEF engineering track).

IN PLAIN WORDS: engineers keep a record of what they tried, what changed,
and whether it met the goals they set BEFORE testing. This program is a menu
you use to record that: your constraints (rules the design must follow),
your success criteria (like "accuracy at least 85%"), and each version
("iteration") you built with its measured results. It then shows how close
each version came to each goal. Judges look for exactly this kind of record.

Interactive, menu-driven log that records:
  * design constraints (limits the solution must respect)
  * success criteria (measurable targets, decided BEFORE testing)
  * prototype iterations (what changed + measured results per criterion)
and calculates how close each iteration came to the success criteria.

Run:  python engineering_log.py
Data is saved to data/engineering_log.json after every change. Standard library only.
"""

import csv
import json
import os
from datetime import date, datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "engineering_log.json")

# Starting values taken from the scoring policy in README_HOW_TO_RUN.md.
# Edit or delete them from the menu; they are only seeded when no log exists yet.
DEFAULT_CONSTRAINTS = [
    {"text": "Official unit is Celsius; temperatures rounded to 0.1 C (half-up) before comparing", "category": "technical"},
    {"text": "Heat Stroke threshold is strictly above 40.5 C (2015 NATA criteria)", "category": "technical"},
    {"text": "Prompt text is locked before any test-set score is seen; experiments use dev cases only", "category": "procedural"},
    {"text": "Web page and Python must give identical answers (check_parity.py)", "category": "technical"},
    {"text": "Uses fictional vignettes only - no real patient data", "category": "safety/ethics"},
    {"text": "Demo must work at the booth without internet (Demo mode)", "category": "practical"},
]

# direction ">=" means higher is better (target is a minimum); "<=" means lower is better (target is a maximum).
DEFAULT_CRITERIA = [
    {"name": "Accuracy (strict exact match)", "unit": "%", "direction": ">=", "target": 85.0},
    {"name": "Heat Stroke recall", "unit": "%", "direction": ">=", "target": 100.0},
    {"name": "Dangerous misses", "unit": "count", "direction": "<=", "target": 0.0},
    {"name": "Run-to-run consistency", "unit": "%", "direction": ">=", "target": 95.0},
    {"name": "Unreadable AI replies", "unit": "count", "direction": "<=", "target": 0.0},
]

# SSEF Junior Division judging rubric (SSEF_Junior_Judging_Rubric.pdf): category, max points, what judges look for.
SSEF_RUBRIC = [
    ("Research Question / Problem", 10,
     "Clear, focused question; testable hypothesis with distinct independent and dependent variables"),
    ("Design & Methodology", 15,
     "Logical procedure, controls clearly identified, adequate trials (minimum 3 recommended)"),
    ("Execution (Data & Coding)", 20,
     "Thorough data collection; code well-structured, functional, commented; graphs show variables accurately"),
    ("Creativity", 20,
     "Real-world problem solved in a novel or clever way; independent problem-solving"),
    ("Poster & Presentation", 35,
     "Logical flow, clear abstract, logbook verification, clear data, engineering choices explained"),
]
MIN_TRIALS = 3


# ---------- storage ----------

def new_log():
    return {
        "project": "Heat illness classifier",
        "created": datetime.now().isoformat(timespec="seconds"),
        "constraints": [dict(c, id=i + 1, added=str(date.today())) for i, c in enumerate(DEFAULT_CONSTRAINTS)],
        "criteria": [dict(c, id=i + 1, added=str(date.today())) for i, c in enumerate(DEFAULT_CRITERIA)],
        "iterations": [],
        "rubric_scores": {},
    }


def load_log(path=None):
    path = path or LOG_PATH
    if not os.path.exists(path):
        return new_log()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_log(log, path=None):
    path = path or LOG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, path)


def next_id(items):
    return max((item["id"] for item in items), default=0) + 1


def find(items, item_id):
    return next((item for item in items if item["id"] == item_id), None)


# ---------- calculations ----------

def criterion_met(criterion, value):
    if criterion["direction"] == ">=":
        return value >= criterion["target"]
    return value <= criterion["target"]


def attainment(criterion, value):
    """Percent of the target reached, capped at 100.

    Higher-is-better: value / target.  Lower-is-better: target / value
    (a target of 0 is all-or-nothing, since any count above 0 misses it).
    """
    target = criterion["target"]
    if criterion["direction"] == ">=":
        if target <= 0:
            return 100.0
        pct = value / target * 100
    else:
        if value <= target:
            return 100.0
        if target <= 0:
            return 0.0
        pct = target / value * 100
    return max(0.0, min(100.0, pct))


def score_iteration(log, iteration):
    """Return (rows, summary) for one iteration.

    rows: one dict per criterion that has a measurement in this iteration.
    summary: criteria met, percent of criteria met, mean attainment.
    """
    rows = []
    for criterion in log["criteria"]:
        key = str(criterion["id"])
        if key not in iteration["results"]:
            continue
        value = iteration["results"][key]
        rows.append({
            "criterion": criterion,
            "value": value,
            "met": criterion_met(criterion, value),
            "attainment": attainment(criterion, value),
        })
    measured = len(rows)
    met = sum(r["met"] for r in rows)
    summary = {
        "measured": measured,
        "total_criteria": len(log["criteria"]),
        "met": met,
        "pct_met": met / measured * 100 if measured else 0.0,
        "mean_attainment": sum(r["attainment"] for r in rows) / measured if measured else 0.0,
    }
    return rows, summary


# ---------- input helpers ----------

def ask(prompt, default=None, allow_blank=False):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return str(default)
        if allow_blank:
            return ""
        print("  Please enter a value.")


def ask_float(prompt, default=None, allow_blank=False):
    while True:
        answer = ask(prompt, default, allow_blank)
        if answer == "" and allow_blank:
            return None
        try:
            return float(answer)
        except ValueError:
            print("  Please enter a number.")


def ask_int(prompt):
    while True:
        try:
            return int(ask(prompt))
        except ValueError:
            print("  Please enter a whole number.")


def ask_choice(prompt, choices, default=None):
    while True:
        answer = ask(f"{prompt} ({'/'.join(choices)})", default)
        if answer in choices:
            return answer
        print(f"  Choose one of: {', '.join(choices)}")


def confirm(prompt):
    return ask(f"{prompt} (y/n)", "n").lower().startswith("y")


def fmt(value):
    return f"{value:g}"


# ---------- constraints ----------

def list_constraints(log):
    print("\nDESIGN CONSTRAINTS")
    if not log["constraints"]:
        print("  (none)")
    for c in log["constraints"]:
        print(f"  [{c['id']}] ({c['category']}) {c['text']}")


def add_constraint(log):
    text = ask("Constraint")
    category = ask("Category (technical/safety/ethics/time/cost/practical/procedural)", "technical")
    log["constraints"].append({"id": next_id(log["constraints"]), "text": text,
                               "category": category, "added": str(date.today())})
    print("  Added.")


def remove_constraint(log):
    list_constraints(log)
    c = find(log["constraints"], ask_int("ID to remove"))
    if c and confirm(f"Remove '{c['text']}'?"):
        log["constraints"].remove(c)
        print("  Removed.")


# ---------- criteria ----------

def list_criteria(log):
    print("\nSUCCESS CRITERIA")
    if not log["criteria"]:
        print("  (none)")
    for c in log["criteria"]:
        print(f"  [{c['id']}] {c['name']}: {c['direction']} {fmt(c['target'])} {c['unit']}")


def add_criterion(log):
    if log["iterations"]:
        print("  Note: criteria should be fixed before testing. Record why you are adding one in the next iteration's notes.")
    name = ask("Criterion name")
    unit = ask("Unit (e.g. %, count, seconds)", "%")
    direction = ask_choice("Higher is better (>=) or lower is better (<=)?", [">=", "<="], ">=")
    target = ask_float("Target value")
    log["criteria"].append({"id": next_id(log["criteria"]), "name": name, "unit": unit,
                            "direction": direction, "target": target, "added": str(date.today())})
    print("  Added.")


def remove_criterion(log):
    list_criteria(log)
    c = find(log["criteria"], ask_int("ID to remove"))
    if c and confirm(f"Remove '{c['name']}'? Its results in past iterations are kept but no longer scored"):
        log["criteria"].remove(c)
        print("  Removed.")


# ---------- iterations ----------

def add_iteration(log):
    if not log["criteria"]:
        print("  Add success criteria first.")
        return
    number = len(log["iterations"]) + 1
    print(f"\nPROTOTYPE ITERATION {number}")
    iteration = {
        "number": number,
        "label": ask("Label (e.g. prompt v2_expanded, test20)", f"Iteration {number}"),
        "date": ask("Date tested (YYYY-MM-DD)", str(date.today())),
        "test_set": ask("Test set (dev/test20/extra30/all50)", "test20"),
        "changes": ask("What changed since the last iteration?", "Initial prototype" if number == 1 else None),
        "trials": ask_int(f"Number of trials/runs (SSEF recommends at least {MIN_TRIALS})"),
        "results": {},
        "error_analysis": "",
        "notes": "",
    }
    if iteration["trials"] < MIN_TRIALS:
        print(f"  Warning: fewer than {MIN_TRIALS} trials - judges may question validity.")
    print("Enter measured results, averaged over the trials (leave blank to skip a criterion):")
    for c in log["criteria"]:
        value = ask_float(f"  {c['name']} ({c['unit']}, target {c['direction']} {fmt(c['target'])})", allow_blank=True)
        if value is not None:
            iteration["results"][str(c["id"])] = value
    iteration["error_analysis"] = ask("Error analysis (which cases failed and why?)", allow_blank=True)
    iteration["notes"] = ask("Observations / problems / next steps", allow_blank=True)
    log["iterations"].append(iteration)
    print()
    show_iteration(log, iteration)


def pick_iteration(log):
    if not log["iterations"]:
        print("  No iterations logged yet.")
        return None
    for it in log["iterations"]:
        print(f"  {it['number']}. {it['label']} ({it['date']}, {it['test_set']})")
    number = ask_int("Iteration number")
    if 1 <= number <= len(log["iterations"]):
        return log["iterations"][number - 1]
    print("  No such iteration.")
    return None


def show_iteration(log, iteration):
    rows, s = score_iteration(log, iteration)
    print(f"ITERATION {iteration['number']}: {iteration['label']}  ({iteration['date']}, set: {iteration['test_set']})")
    print(f"  Changes: {iteration['changes']}")
    trials = iteration.get("trials")
    if trials is not None:
        print(f"  Trials: {trials}" + ("" if trials >= MIN_TRIALS else f"  (below SSEF minimum of {MIN_TRIALS})"))
    print(f"  {'Criterion':<34}{'Result':>10}{'Target':>14}{'Attain':>9}  Met")
    for r in rows:
        c = r["criterion"]
        print(f"  {c['name']:<34}{fmt(r['value']):>10}{c['direction'] + ' ' + fmt(c['target']):>14}"
              f"{r['attainment']:>8.1f}%  {'YES' if r['met'] else 'no'}")
    print(f"  Criteria met: {s['met']}/{s['measured']} measured = {s['pct_met']:.1f}%"
          f"  (of {s['total_criteria']} total criteria)")
    print(f"  Mean attainment: {s['mean_attainment']:.1f}%")
    if iteration.get("error_analysis"):
        print(f"  Error analysis: {iteration['error_analysis']}")
    if iteration.get("notes"):
        print(f"  Notes: {iteration['notes']}")


def view_iteration(log):
    iteration = pick_iteration(log)
    if iteration:
        print()
        show_iteration(log, iteration)


def edit_result(log):
    iteration = pick_iteration(log)
    if not iteration:
        return
    list_criteria(log)
    c = find(log["criteria"], ask_int("Criterion ID"))
    if not c:
        print("  No such criterion.")
        return
    old = iteration["results"].get(str(c["id"]))
    value = ask_float("New value", old)
    reason = ask("Reason for the correction")
    iteration["results"][str(c["id"])] = value
    iteration["notes"] = (iteration.get("notes", "") + f" [Corrected {c['name']}: {old} -> {fmt(value)} on "
                          f"{date.today()}: {reason}]").strip()
    print("  Updated (correction recorded in notes).")


# ---------- reports ----------

def progress_report(log):
    if not log["iterations"]:
        print("  No iterations logged yet.")
        return
    print("\nPROGRESS ACROSS ITERATIONS")
    print(f"  {'#':<4}{'Label':<30}{'Met':>8}{'% met':>9}{'Mean attain':>13}")
    for it in log["iterations"]:
        _, s = score_iteration(log, it)
        print(f"  {it['number']:<4}{it['label'][:29]:<30}{str(s['met']) + '/' + str(s['measured']):>8}"
              f"{s['pct_met']:>8.1f}%{s['mean_attainment']:>12.1f}%")

    print("\n  Per criterion (value by iteration):")
    for c in log["criteria"]:
        values = []
        for it in log["iterations"]:
            v = it["results"].get(str(c["id"]))
            values.append("-" if v is None else fmt(v) + ("*" if criterion_met(c, v) else ""))
        print(f"  {c['name']:<34} " + "  ".join(f"{v:>7}" for v in values))
    print("  (* = target met)")

    best = max(log["iterations"], key=lambda it: score_iteration(log, it)[1]["mean_attainment"])
    print(f"\n  Best iteration so far: {best['number']} ({best['label']})")


def rubric_evidence(log):
    """Things the log itself can show a judge, per rubric category."""
    its = log["iterations"]
    low_trials = [it["number"] for it in its if it.get("trials", 0) < MIN_TRIALS]
    no_error_analysis = [it["number"] for it in its if not it.get("error_analysis")]
    evidence = {
        "Design & Methodology": [
            f"{len(log['constraints'])} design constraints logged",
            f"{len(log['criteria'])} success criteria with targets",
            "All iterations have >= 3 trials" if its and not low_trials
            else f"Iterations below {MIN_TRIALS} trials: {low_trials or 'no iterations yet'}",
        ],
        "Execution (Data & Coding)": [
            f"{len(its)} prototype iterations recorded",
            "Error analysis written for every iteration" if its and not no_error_analysis
            else f"Iterations missing error analysis: {no_error_analysis or 'no iterations yet'}",
        ],
        "Poster & Presentation": [
            "Dated logbook entries with corrections recorded (export option 11 for the board)",
        ],
    }
    return evidence


def rubric_check(log):
    scores = log.setdefault("rubric_scores", {})
    evidence = rubric_evidence(log)
    print("\nSSEF JUNIOR RUBRIC SELF-CHECK")
    for name, max_pts, expects in SSEF_RUBRIC:
        print(f"\n  {name} (max {max_pts})")
        print(f"    Judges look for: {expects}")
        for line in evidence.get(name, []):
            print(f"    - {line}")
        if name in scores:
            print(f"    Current self-score: {fmt(scores[name])}/{max_pts}")
    if confirm("\nUpdate self-scores now?"):
        for name, max_pts, _ in SSEF_RUBRIC:
            while True:
                value = ask_float(f"  {name} (0-{max_pts})", scores.get(name))
                if 0 <= value <= max_pts:
                    scores[name] = value
                    break
                print(f"  Must be between 0 and {max_pts}.")
    rubric_summary(log)


def rubric_summary(log):
    scores = log.get("rubric_scores", {})
    if not scores:
        return
    total_max = sum(max_pts for _, max_pts, _ in SSEF_RUBRIC)
    total = sum(scores.get(name, 0) for name, _, _ in SSEF_RUBRIC)
    print(f"\n  {'Category':<30}{'Score':>8}{'%':>8}")
    for name, max_pts, _ in SSEF_RUBRIC:
        s = scores.get(name, 0)
        print(f"  {name:<30}{fmt(s) + '/' + str(max_pts):>8}{s / max_pts * 100:>7.0f}%")
    print(f"  {'TOTAL':<30}{fmt(total) + '/' + str(total_max):>8}{total / total_max * 100:>7.0f}%")
    weakest = min(SSEF_RUBRIC, key=lambda r: scores.get(r[0], 0) / r[1])
    print(f"  Most points to gain: {weakest[0]}")


def export_files(log):
    folder = os.path.dirname(LOG_PATH)
    os.makedirs(folder, exist_ok=True)

    csv_path = os.path.join(folder, "engineering_log_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "label", "date", "test_set", "criterion", "unit",
                         "direction", "target", "value", "met", "attainment_pct"])
        for it in log["iterations"]:
            for r in score_iteration(log, it)[0]:
                c = r["criterion"]
                writer.writerow([it["number"], it["label"], it["date"], it["test_set"], c["name"], c["unit"],
                                 c["direction"], c["target"], r["value"], r["met"], round(r["attainment"], 1)])

    md_path = os.path.join(folder, "engineering_log.md")
    lines = [f"# Engineering log: {log['project']}", "", "## Design constraints", ""]
    lines += [f"- ({c['category']}) {c['text']}" for c in log["constraints"]]
    lines += ["", "## Success criteria", "", "| Criterion | Target |", "|---|---|"]
    lines += [f"| {c['name']} | {c['direction']} {fmt(c['target'])} {c['unit']} |" for c in log["criteria"]]
    lines += ["", "## Iterations", ""]
    for it in log["iterations"]:
        rows, s = score_iteration(log, it)
        lines += [f"### Iteration {it['number']}: {it['label']}", "",
                  f"Date: {it['date']} | Test set: {it['test_set']}", "",
                  f"Changes: {it['changes']}", "",
                  "| Criterion | Result | Target | Attainment | Met |", "|---|---|---|---|---|"]
        if it.get("trials") is not None:
            lines[-2:-2] = [f"Trials: {it['trials']}", ""]
        lines += [f"| {r['criterion']['name']} | {fmt(r['value'])} | {r['criterion']['direction']} "
                  f"{fmt(r['criterion']['target'])} | {r['attainment']:.1f}% | {'Yes' if r['met'] else 'No'} |"
                  for r in rows]
        lines += ["", f"Criteria met: {s['met']}/{s['measured']} ({s['pct_met']:.1f}%). "
                      f"Mean attainment: {s['mean_attainment']:.1f}%.", ""]
        if it.get("error_analysis"):
            lines += [f"Error analysis: {it['error_analysis']}", ""]
        if it.get("notes"):
            lines += [f"Notes: {it['notes']}", ""]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Wrote {csv_path}")
    print(f"  Wrote {md_path}")


# ---------- menu ----------

MENU = [
    ("1", "List design constraints", list_constraints, False),
    ("2", "Add design constraint", add_constraint, True),
    ("3", "Remove design constraint", remove_constraint, True),
    ("4", "List success criteria", list_criteria, False),
    ("5", "Add success criterion", add_criterion, True),
    ("6", "Remove success criterion", remove_criterion, True),
    ("7", "Log a prototype iteration (test results)", add_iteration, True),
    ("8", "View an iteration", view_iteration, False),
    ("9", "Correct a recorded result", edit_result, True),
    ("10", "Progress report across iterations", progress_report, False),
    ("11", "Export CSV + Markdown (for the board / logbook)", export_files, False),
    ("12", "SSEF Junior rubric self-check", rubric_check, True),
]


def main():
    first_run = not os.path.exists(LOG_PATH)
    log = load_log()
    if first_run:
        save_log(log)
        print(f"Created {LOG_PATH} with starting constraints and criteria from the README scoring policy.")
    actions = {key: (fn, changes) for key, _, fn, changes in MENU}
    while True:
        print(f"\n=== Engineering log: {log['project']} "
              f"({len(log['constraints'])} constraints, {len(log['criteria'])} criteria, "
              f"{len(log['iterations'])} iterations) ===")
        for key, label, _, _ in MENU:
            print(f"  {key:>2}. {label}")
        print("   q. Quit")
        choice = input("> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice not in actions:
            print("  Unknown option.")
            continue
        fn, changes = actions[choice]
        try:
            fn(log)
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
            continue
        if changes:
            save_log(log)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
