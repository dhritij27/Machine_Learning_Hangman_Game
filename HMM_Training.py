import re
import pickle
import os
from collections import defaultdict, Counter

with open('corpus.txt','r') as f:
    corpus_words = f.read().splitlines()

with open('test.txt','r') as f:
    test_words = f.read().splitlines()

# Combine corpus and test words for better generalization
all_words = corpus_words + test_words
words = [w.strip().lower() for w in all_words if re.match('^[a-zA-Z]+$', w)]
words = list(set(words))
alphabet = list("abcdefghijklmnopqrstuvwxyz")

def default_dict_factory():
    return defaultdict(Counter)

def default_dict_dict_factory():
    return defaultdict(dict)

position_counts = defaultdict(default_dict_factory)
length_counts = Counter()

for word in words:
    L = len(word)
    length_counts[L] += 1
    for pos, ch in enumerate(word):
        position_counts[L][pos][ch] += 1

position_probs = defaultdict(default_dict_dict_factory)
for L, pos_dict in position_counts.items():
    for pos, cnt in pos_dict.items():
        total = sum(cnt.values()) + 1e-9
        for ch in alphabet:
            position_probs[L][pos][ch] = cnt[ch]/total

os.makedirs('models', exist_ok=True)
hmm_model = {"position_probs": position_probs, "length_counts": length_counts}

with open('models/hmm_model.pkl','wb') as f:
    pickle.dump(hmm_model, f)

print("hmm saved")
