"""Interactive demo of the heat illness classifier.

IN PLAIN WORDS: a text-only version of the app for the terminal. Type a
description and a temperature, and it prints the label and the AI's two
answers. Needs GROQ_API_KEY set. Run:  py demo.py

Run this and type in a symptom description and a temperature to get a
live classification from the pipeline -- useful for showing judges how
the system actually works, instead of only showing them a results table.

This reuses classify_heat_illness() from full_classifier_groq.py rather
than duplicating it, so it's always testing the exact same logic your
evaluation results are based on.
"""

import full_classifier_groq as classifier


def main():
    print("=" * 60)
    print("Heat Illness Classifier -- Interactive Demo")
    print("=" * 60)
    print("Describe a scenario and a temperature; the pipeline will")
    print("classify it live using the LLM. Type 'quit' at any prompt")
    print("to exit.\n")

    while True:
        symptom_text = input("Symptom description: ").strip()
        if symptom_text.lower() in ("quit", "exit", ""):
            break

        temp_input = input("Temperature (Celsius): ").strip()
        if temp_input.lower() in ("quit", "exit"):
            break
        try:
            temperature_c = float(temp_input)
        except ValueError:
            print("  Please enter a number for temperature, e.g. 39.5\n")
            continue

        print("  Thinking...\n")
        label, cns_flag, heat_flag, cns_reason, heat_reason = classifier.classify_heat_illness(
            symptom_text, temperature_c
        )

        print(f"  Classification: {label}")
        print(f"  CNS dysfunction detected: {'Yes' if cns_flag else 'No'} -- {cns_reason.strip()}")
        print(f"  Heat symptoms detected:   {'Yes' if heat_flag else 'No'} -- {heat_reason.strip()}")
        print("-" * 60 + "\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
