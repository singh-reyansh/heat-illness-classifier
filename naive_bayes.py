"""
A Naive Bayes text classifier, written from scratch (no machine learning
libraries). It powers the chatbot: it reads a question and guesses which
topic it is about.

How it works, in two steps:
  TRAINING   = counting. For each topic, count how often every word appears
               in that topic's example questions.
  PREDICTING = multiplying fractions. For each topic, multiply
               (how common the topic is) x (the fraction for each word in the
               question). The topic with the biggest score wins.

compare_with_sklearn.py checks that this gives the same answers as
scikit-learn's MultinomialNB.
"""
import re


def tokenize(text):
    """Split a sentence into lowercase words.
    "Can I give him water?" -> ["can", "i", "give", "him", "water"]"""
    return re.findall(r"[a-z0-9']+", text.lower())


class NaiveBayes:
    def __init__(self, smoothing=1):
        # smoothing = the number added to every word count ("add-one
        # smoothing") so that no fraction is ever 0.
        self.smoothing = smoothing
        self.topics = []           # every topic name
        self.question_counts = {}  # topic -> how many training questions it has
        self.word_counts = {}      # topic -> {word: how many times it appeared}
        self.total_words = {}      # topic -> total number of words in that topic
        self.vocabulary = set()    # every different word seen in training

    def train(self, questions, topics):
        """questions[i] is an example question; topics[i] is its topic."""
        for question, topic in zip(questions, topics):
            if topic not in self.word_counts:
                self.topics.append(topic)
                self.question_counts[topic] = 0
                self.word_counts[topic] = {}
                self.total_words[topic] = 0

            self.question_counts[topic] += 1
            for word in tokenize(question):
                self.word_counts[topic][word] = self.word_counts[topic].get(word, 0) + 1
                self.total_words[topic] += 1
                self.vocabulary.add(word)

    def word_fraction(self, word, topic):
        """How likely this word is to show up in a question about this topic,
        with add-one smoothing: (count + 1) / (total words + vocabulary size)."""
        count = self.word_counts[topic].get(word, 0)
        return (count + self.smoothing) / (self.total_words[topic] + self.smoothing * len(self.vocabulary))

    def predict_all(self, question):
        """Returns {topic: chance from 0 to 1} for every topic, adding up to 1."""
        # Words never seen in training are skipped: the model knows nothing about them.
        words = [w for w in tokenize(question) if w in self.vocabulary]
        total_questions = sum(self.question_counts.values())

        scores = {}
        for topic in self.topics:
            score = self.question_counts[topic] / total_questions  # the "prior": how common this topic is
            for word in words:
                score *= self.word_fraction(word, topic)
            scores[topic] = score

        # Turn the scores into percentages by dividing each by the total.
        total = sum(scores.values())
        if total == 0:  # only happens with extremely long questions (numbers too tiny for the computer)
            return {topic: 1 / len(self.topics) for topic in self.topics}
        return {topic: score / total for topic, score in scores.items()}

    def predict(self, question):
        """Returns (best topic, how sure from 0 to 1)."""
        chances = self.predict_all(question)
        best = max(chances, key=chances.get)
        return best, chances[best]


if __name__ == "__main__":
    # The worked example from our explanation. Expect: water, about 92%.
    model = NaiveBayes()
    model.train(
        ["can he drink water", "can i give water", "should he drink",
         "why heat stroke", "why is it heat stroke", "why did it say stroke"],
        ["water", "water", "water", "why", "why", "why"],
    )
    topic, chance = model.predict("can he drink it")
    print(f"'can he drink it' -> {topic} ({chance:.1%} sure)")
