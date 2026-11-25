"""
An older, simpler web app made with Streamlit (opens at localhost:8501).

IN PLAIN WORDS: Streamlit turns a Python script into a web page: every
st.something() line draws one thing on the page (a title, a text box, a
button). Tab 1 classifies one description; tab 2 runs all 20 test vignettes
and shows the accuracy. It calls the same classifier as everything else.

The main app is now server.py + classifier_flask.html (more features: demo
mode, chatbot, first-aid steps). This one is kept as a simple backup.
Run:  streamlit run app.py   (needs GROQ_API_KEY set first)
"""

import streamlit as st

import full_classifier_groq as classifier
from vignette_sets import load_set
from stats_utils import confusion_matrix_report, wilson_confidence_interval

st.set_page_config(page_title="Heat Illness Classifier", page_icon=":thermometer:")

st.title("AI-Assisted Heat Illness Classifier")
st.caption(
    "Combines an LLM (for detecting CNS dysfunction and heat symptoms from a "
    "written description) with a rule based on NATA/ACSM classification criteria."
)

tab1, tab2 = st.tabs(["Try it yourself", "Run full evaluation"])

# ---------------------------------------------------------------------------
# Tab 1: single live classification
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Classify a scenario")

    symptom_text = st.text_area(
        "Symptom description",
        placeholder=(
            "e.g. A runner is confused about what day it is and can't follow "
            "simple instructions, though she is still walking and talking."
        ),
        height=100,
    )
    temperature_c = st.number_input(
        "Temperature (Celsius)", min_value=30.0, max_value=45.0, value=38.0, step=0.1
    )

    if st.button("Classify", type="primary"):
        if not symptom_text.strip():
            st.warning("Please enter a symptom description first.")
        else:
            with st.spinner("Classifying..."):
                label, cns_flag, heat_flag, cns_reason, heat_reason = classifier.classify_heat_illness(
                    symptom_text, temperature_c
                )

            if label == "Heat Stroke":
                st.error(f"Classification: **{label}**")
            elif label == "Uncertain — flag for review":
                st.warning(f"Classification: **{label}**")
            elif label == "Heat Exhaustion":
                st.info(f"Classification: **{label}**")
            else:
                st.success(f"Classification: **{label}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("CNS dysfunction detected", "Yes" if cns_flag else "No")
                st.caption(cns_reason.strip())
            with col2:
                st.metric("Heat symptoms detected", "Yes" if heat_flag else "No")
                st.caption(heat_reason.strip())

# ---------------------------------------------------------------------------
# Tab 2: run the full vignette evaluation, live, from the browser
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Run the full 20-vignette evaluation")
    st.write(
        "Runs every vignette in `data/vignettes_test20.json` through the classifier and "
        "reports accuracy, a 95% confidence interval, and a confusion matrix "
        "-- the same evaluation the terminal scripts run, shown here instead."
    )

    if st.button("Run evaluation"):
        vignettes = load_set("test20")

        n = len(vignettes)
        progress = st.progress(0, text="Starting...")
        rows = []
        correct = 0

        for i, v in enumerate(vignettes):
            progress.progress(i / n, text=f"Classifying {v['id']}...")
            predicted, cns_flag, heat_flag, cns_reason, heat_reason = classifier.classify_heat_illness(
                v["symptom_text"], v["temperature_c"]
            )
            is_match = predicted == v["true_label"]
            correct += is_match
            rows.append({
                "ID": v["id"],
                "Predicted": predicted,
                "True Label": v["true_label"],
                "Match": "correct" if is_match else "WRONG",
            })

        progress.progress(1.0, text="Done.")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        accuracy = correct / n * 100
        ci_low, ci_high = wilson_confidence_interval(correct, n)

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.1f}%", f"{correct}/{n} correct")
        col2.metric("95% confidence interval", f"{ci_low * 100:.1f}% - {ci_high * 100:.1f}%")

        with st.expander("Confusion matrix / precision / recall (see terminal-style output)"):
            import io
            import contextlib

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                confusion_matrix_report(
                    true_labels=[r["True Label"] for r in rows],
                    predicted_labels=[r["Predicted"] for r in rows],
                )
            st.code(buffer.getvalue(), language=None)
