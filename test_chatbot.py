"""
Tests the chatbot's Naive Bayes model on the TEST questions in
chatbot_data.json -- questions it never saw during training.

Prints:
  1. accuracy (with a 95% confidence range)
  2. every question it got wrong
  3. a confusion matrix (which topics it mixes up)
  4. how different "I'm not sure" cutoffs would change things

Run:  py test_chatbot.py

Honest-testing rule: if you change the training questions to fix mistakes,
write NEW test questions before reporting a final score.
"""
import chatbot
from stats_utils import confusion_matrix_report, wilson_confidence_interval

data = chatbot.load_data()
model = chatbot.build_model(data)

true_topics, predicted_topics, confidences = [], [], []
for topic, info in data.items():
    for question in info["test"]:
        predicted, confidence = model.predict(question)
        true_topics.append(topic)
        predicted_topics.append(predicted)
        confidences.append(confidence)

n = len(true_topics)
correct = sum(t == p for t, p in zip(true_topics, predicted_topics))
low, high = wilson_confidence_interval(correct, n)
print(f"Trained on {len(model.vocabulary)} different words from {sum(model.question_counts.values())} questions.")
print(f"\n1. ACCURACY: {correct}/{n} = {correct / n:.1%}   (95% range: {low:.1%} to {high:.1%})\n")

print("2. MISTAKES:")
i = 0
for topic, info in data.items():
    for question in info["test"]:
        if true_topics[i] != predicted_topics[i]:
            print(f"   {question!r}: should be {true_topics[i]}, got {predicted_topics[i]} ({confidences[i]:.0%} sure)")
        i += 1
if correct == n:
    print("   none")
print()

print("3. ", end="")
confusion_matrix_report(true_topics, predicted_topics)
print()

print("4. CUTOFF TEST (below the cutoff, the chatbot says \"I'm not sure\" instead of answering):")
print(f"   {'Cutoff':<8}{'Answered':<11}{'Right when answered':<22}{'Wrong answers given'}")
for cutoff in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    answered = [(t, p) for t, p, c in zip(true_topics, predicted_topics, confidences) if c >= cutoff]
    right = sum(t == p for t, p in answered)
    wrong = len(answered) - right
    rate = f"{right / len(answered):.0%}" if answered else "-"
    print(f"   {cutoff:<8.0%}{len(answered):>3}/{n:<7}{rate:<22}{wrong}")
print(f"\n(The chatbot currently uses a cutoff of {chatbot.CONFIDENCE_CUTOFF:.0%}.)")
