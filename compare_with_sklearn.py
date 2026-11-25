"""
Checks the from-scratch Naive Bayes (naive_bayes.py) against scikit-learn's
MultinomialNB, a professional library. Both are trained on the same questions
and asked about every training and test question. If the topics AND the
percentages match, the from-scratch math is correct.

Run:  py compare_with_sklearn.py
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import chatbot

data = chatbot.load_data()
train_questions, train_topics = chatbot.training_examples(data)
all_questions = train_questions + [q for info in data.values() for q in info["test"]]

# 1. My model
mine = chatbot.build_model(data)

# 2. scikit-learn's model, set up to work the same way:
#    - the same word-splitting rule as tokenize() in naive_bayes.py
#    - alpha=1 is the same add-one smoothing
vectorizer = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9']+")
word_counts = vectorizer.fit_transform(train_questions)
sk_model = MultinomialNB(alpha=1)
sk_model.fit(word_counts, train_topics)

sk_topics = sk_model.predict(vectorizer.transform(all_questions))
sk_chances = sk_model.predict_proba(vectorizer.transform(all_questions))

same_topic = 0
ties = 0
biggest_difference = 0.0
for i, question in enumerate(all_questions):
    my_chances = mine.predict_all(question)
    my_topic = max(my_chances, key=my_chances.get)
    if my_topic == sk_topics[i]:
        same_topic += 1
    elif abs(my_chances[my_topic] - my_chances[sk_topics[i]]) < 1e-12:
        # Both topics have exactly the same score (usually because the question
        # has no words the model knows), and the two programs break the tie
        # differently. That is not a bug, so it counts as a match.
        same_topic += 1
        ties += 1
        print(f"TIE: {question!r} -> no known words, every topic scores the same")
    else:
        print(f"DIFFERENT: {question!r} -> mine: {my_topic}, scikit-learn: {sk_topics[i]}")
    for j, topic in enumerate(sk_model.classes_):
        biggest_difference = max(biggest_difference, abs(my_chances[topic] - sk_chances[i][j]))

print(f"Same topic on {same_topic} of {len(all_questions)} questions ({ties} were exact ties).")
print(f"Biggest difference in any percentage: {biggest_difference * 100:.8f} percentage points")
if same_topic == len(all_questions) and biggest_difference < 1e-9:
    print("MATCH: the from-scratch model gives the same answers as scikit-learn.")
else:
    print("NO MATCH: look at the differences above to find the bug.")
