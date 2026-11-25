"""The decision rule -- the ONE place it is defined.

IN PLAIN WORDS (picture: docs/images/02_decision_rule.svg)
  The AI answers two yes/no questions: "Is there a CNS sign?" and "Is there
  a heat symptom?". This file turns those two answers plus the temperature
  into one of four labels, checked in this order:
    1. CNS sign?                   -> Heat Stroke
    2. Temperature above 40.5 C?   -> Uncertain (flag for a person to review)
    3. Heat symptom?               -> Heat Exhaustion
    4. none of the above           -> Not Heat Illness
  Keeping the medical decision in a fixed rule (instead of letting the AI
  pick the label) makes the tool predictable, testable, and explainable.

The AI never makes the medical decision. It only answers two yes/no
questions about the written description (CNS dysfunction? heat symptom?).
Everything below is the fixed rule that turns those two answers plus the
temperature into a label.

Review changes built in here:
  * Item 12 -- official unit is Celsius. The threshold is "strictly above
    40.5 C". Temperatures are rounded to 0.1 C first (half-up), so exactly
    40.5 C is NOT above the threshold, and 105.0 F (= 40.56 C -> 40.6 C) IS.
  * Item 3  -- the rule assumes a CORE (rectal) temperature. When the reading
    was taken another way, the result carries a caveat saying the reading
    cannot rule out heat stroke. The label itself is not changed.
  * Item 4  -- CNS signs alone give "Heat Stroke" on purpose (safety first:
    cool first, transport second). The result says so explicitly.
  * Item 7  -- scoring of the "Uncertain" label is defined here too, BEFORE
    any results are looked at.
"""
from decimal import Decimal, ROUND_HALF_UP

TEMP_THRESHOLD_C = Decimal("40.5")

LABEL_HEAT_STROKE = "Heat Stroke"
LABEL_HEAT_EXHAUSTION = "Heat Exhaustion"
LABEL_UNCERTAIN = "Uncertain — flag for review"
LABEL_NOT_HEAT_ILLNESS = "Not Heat Illness — no thermal stress indicated"
LABEL_UNPARSEABLE = "Unparseable — no answer"   # never produced by the rule; used only when the AI's reply can't be read

ALL_LABELS = [LABEL_HEAT_STROKE, LABEL_HEAT_EXHAUSTION, LABEL_UNCERTAIN, LABEL_NOT_HEAT_ILLNESS]

# Methods NATA says are not valid or reliable for core temperature after
# intense exercise in the heat. Rectal is the accepted field standard.
NON_RECTAL_SOURCES = {"oral", "tympanic", "forehead", "axillary"}
SOURCE_LABELS = {
    "": "not specified", "oral": "oral", "tympanic": "tympanic (ear)",
    "forehead": "forehead / temporal", "axillary": "axillary (armpit)", "rectal": "rectal",
}


def fahrenheit_to_celsius(temp_f):
    return (float(temp_f) - 32.0) * 5.0 / 9.0


def normalize_temp_c(temp_c):
    """Round to 0.1 C, half-up. Uses Decimal on the printed value so that
    e.g. 40.55 rounds to 40.6 (plain float round() would give 40.5).

    Why the fuss? Computers store decimals in binary, so 40.55 is really
    40.549999... and normal rounding would give 40.5. Decimal does the
    rounding the way you learned in school."""
    return Decimal(repr(round(float(temp_c), 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def is_above_threshold(temp_c):
    return normalize_temp_c(temp_c) > TEMP_THRESHOLD_C


def apply_rule(cns_flag, heat_flag, temp_c):
    """The four-branch rule. CNS dysfunction always wins; otherwise a
    temperature strictly above 40.5 C with no CNS signs is Uncertain;
    otherwise a heat symptom means Heat Exhaustion; otherwise Not Heat
    Illness."""
    if cns_flag == 1:                  # 1. brain/nerve sign -> emergency, no matter the temperature
        return LABEL_HEAT_STROKE
    if is_above_threshold(temp_c):     # 2. very hot but no CNS sign -> doesn't fit cleanly, a person decides
        return LABEL_UNCERTAIN
    if heat_flag == 1:                 # 3. heat symptoms, thinking normally -> heat exhaustion
        return LABEL_HEAT_EXHAUSTION
    return LABEL_NOT_HEAT_ILLNESS      # 4. nothing found


def build_caveats(label, temp_source, cns_flag=0, context_text="", context_exposure_flag=None):
    """Plain-language notes shown with a result. They never change the label.

    context_exposure_flag: 1/0 answer to "does the context mention exertion or
    heat exposure?" -- or None if no context was given / not asked."""
    caveats = []
    source = (temp_source or "").strip().lower()

    if source != "rectal":
        if source == "":
            how = "The temperature method was not specified."
        else:
            how = f"The temperature was taken by a non-rectal method ({SOURCE_LABELS.get(source, source)})."
        if label in (LABEL_HEAT_EXHAUSTION, LABEL_NOT_HEAT_ILLNESS):
            caveats.append(
                f"{how} NATA guidance says only a rectal temperature is valid after intense exercise in the heat, "
                "so this reading cannot rule out heat stroke. Get a rectal temperature if possible and keep watching for CNS signs."
            )
        elif label == LABEL_UNCERTAIN:
            caveats.append(
                f"{how} Non-rectal readings usually read lower than core temperature, so the true value may be even higher. "
                "Get a rectal temperature and have a clinician review this case."
            )
        else:  # heat stroke
            caveats.append(
                f"{how} Non-rectal readings can under-read, so do not wait for a higher number before treating."
            )

    if label == LABEL_HEAT_STROKE and cns_flag == 1:
        caveats.append(
            "Classified as Heat Stroke from CNS signs alone, on purpose: when in doubt, treat as an emergency (cool first, transport second). "
            "The NATA definition uses both CNS dysfunction and a core temperature above 40.5 C (104.9 F); this tool does not wait for the temperature "
            "because field temperature readings are often unreliable."
        )
        if (context_text or "").strip() and context_exposure_flag == 0:
            caveats.append(
                "The 'what was the patient doing' note does not mention exercise or heat exposure. Other causes of confusion "
                "(for example a concussion or low blood sugar) can look like this. Treat it as an emergency anyway."
            )
        elif not (context_text or "").strip():
            caveats.append(
                "No heat or exercise context was given. This tool does not check for other causes of confusion "
                "(for example a concussion or low blood sugar); it is only meant for people who were exercising or exposed to heat."
            )
    return caveats


# ---------------------------------------------------------------------------
# Scoring policy (item 7) -- decided before running anything.
#   PRIMARY metric   : strict exact match. "Uncertain" is its own class, so a
#                      case whose true label is Heat Stroke but which the tool
#                      flags as Uncertain is WRONG.
#   SAFETY metric    : a "dangerous miss" is a true Heat Stroke predicted as
#                      Heat Exhaustion or Not Heat Illness (or unparseable).
#                      A flag for review is NOT a dangerous miss, because a
#                      human then looks at the case.
#   Unparseable      : if the AI's reply still can't be read after the
#                      retries, the case is counted WRONG and reported.
# ---------------------------------------------------------------------------
def is_exact_match(true_label, predicted_label):
    return true_label == predicted_label


def is_dangerous_miss(true_label, predicted_label):
    return true_label == LABEL_HEAT_STROKE and predicted_label not in (LABEL_HEAT_STROKE, LABEL_UNCERTAIN)
