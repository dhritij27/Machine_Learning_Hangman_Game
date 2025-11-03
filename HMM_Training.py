import re
import numpy as np
import pickle
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

with open('../data/corpus.txt', 'r') as f:
    words = f.read().splitlines()

words = [w.strip().lower() for w in words if re.match('^[a-zA-Z]+$', w)]
words = list(set(words))
print(f"Clean words: {len(words)}")


words_by_length = defaultdict(list)
for w in words:
    words_by_length[len(w)].append(w)

print("Example group:", list(words_by_length.keys())[:10])


alphabet = list("abcdefghijklmnopqrstuvwxyz")
alpha_size = len(alphabet)

# Transition: P(next_letter | current_letter)
transition_counts = {a: Counter() for a in alphabet}
emission_counts = Counter()

for word in words:
    for i in range(len(word)-1):
        transition_counts[word[i]][word[i+1]] += 1
    for ch in word:
        emission_counts[ch] += 1


transition_probs = {}
for a in alphabet:
    total = sum(transition_counts[a].values()) + 1e-9
    transition_probs[a] = {b: transition_counts[a][b]/total for b in alphabet}

emission_probs = {ch: emission_counts[ch]/sum(emission_counts.values()) for ch in alphabet}


hmm_model = {
    "transition_probs": transition_probs,
    "emission_probs": emission_probs
}

with open('../models/hmm_model.pkl', 'wb') as f:
    pickle.dump(hmm_model, f)

print("HMM Model saved successfully!")


def predict_letter_probabilities(pattern, guessed_letters):
    """
    pattern: e.g. '_a__e_'
    guessed_letters: set of already guessed letters
    returns: probability dict for each letter
    """
    available_letters = [a for a in alphabet if a not in guessed_letters]
    letter_scores = {a: 0 for a in available_letters}

   
    for i, ch in enumerate(pattern):
        if ch != "_":
            for a in available_letters:
                letter_scores[a] += transition_probs[ch].get(a, 0)

    
    total = sum(letter_scores.values()) + 1e-9
    probs = {a: letter_scores[a]/total for a in available_letters}
    return probs

print(predict_letter_probabilities("_a__e_", {'a', 'e', 's', 't'})[:5])
