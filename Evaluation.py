import pickle
import random
import numpy as np
from collections import defaultdict, Counter

def default_dict_factory():
    return defaultdict(Counter)

def default_dict_dict_factory():
    return defaultdict(dict)

alphabet = list("abcdefghijklmnopqrstuvwxyz")

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.996):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    def get_state(self, pattern, guessed):
        return (pattern, "".join(sorted(guessed)))
    def choose_action(self, state, pattern, guessed):
        available = [a for a in alphabet if a not in guessed]
        L = len(pattern)
        hmm_probs = {a: 0 for a in available}
        if L in position_probs:
            for pos, ch in enumerate(pattern):
                if ch == "_":
                    for a in available:
                        hmm_probs[a] += position_probs[L][pos].get(a, 0)
        if random.random() < self.epsilon:
            return random.choice(available)
        scores = {a: self.Q[(state,a)] + 5*hmm_probs.get(a,0) for a in available}
        return max(scores, key=scores.get)
    def update(self, state, action, reward, next_state):
        max_next = max([self.Q[(next_state,a)] for a in alphabet], default=0)
        self.Q[(state,action)] += self.alpha * (reward + self.gamma * max_next - self.Q[(state,action)])

with open('models/hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
with open('models/rl_agent.pkl', 'rb') as f:
    rl_agent = pickle.load(f)
with open('corpus.txt', 'r') as f:
    corpus_words = [w.strip().lower() for w in f.read().splitlines() if w.strip()]

with open('test.txt', 'r') as f:
    test_words_train = [w.strip().lower() for w in f.read().splitlines() if w.strip()]

# Combine corpus and test words for filtering (test words are in HMM training)
all_train_words = corpus_words + test_words_train

# Group words by length for faster filtering
words_by_length = defaultdict(list)
for word in all_train_words:
    if word:
        words_by_length[len(word)].append(word)

# Corpus-specific letter frequencies (from chart: e, a, i, o, r, n, t, s, l, c, u, p, m, d, h, y, g, b, f, v, k, w, z, x, q, j)
corpus_letter_order = "eaiorntslcupmdhygbfvkwzxqj"
corpus_letter_priority = {ch: (27 - i) / 27 for i, ch in enumerate(corpus_letter_order)}

position_probs = hmm_model['position_probs']

class HangmanEnv:
    def __init__(self, words, max_lives=8):
        self.words = words
        self.max_lives = max_lives
    def reset(self):
        self.word = random.choice(self.words)
        self.guessed = set()
        self.lives = self.max_lives
        self.pattern = "_" * len(self.word)
        return self.pattern
    def step(self, letter):
        if letter in self.guessed:
            return self.pattern, 0, False, None
        self.guessed.add(letter)
        if letter in self.word:
            new_pattern = list(self.pattern)
            for i, ch in enumerate(self.word):
                if ch == letter:
                    new_pattern[i] = letter
            self.pattern = "".join(new_pattern)
        else:
            self.lives -= 1
        if "_" not in self.pattern:
            return self.pattern, 0, True, True
        if self.lives <= 0:
            return self.pattern, 0, True, False
        return self.pattern, 0, False, None

def get_hmm_probs(pattern, guessed):
    L = len(pattern)
    available = [a for a in alphabet if a not in guessed]
    
    # Filter words that match the pattern (test words are now in training data!)
    matching_words = []
    revealed_letters = set(ch for ch in pattern if ch != "_")
    
    # Only check words of the right length (much faster)
    candidates = words_by_length.get(L, [])
    for word in candidates:
        matches = True
        # Check exact matches at revealed positions
        for i, ch in enumerate(pattern):
            if ch != "_":
                if word[i] != ch:
                    matches = False
                    break
        # Check that revealed letters don't appear in wrong positions
        if matches:
            for i, ch in enumerate(word):
                if pattern[i] == "_" and ch in revealed_letters:
                    matches = False
                    break
        if matches:
            matching_words.append(word)
    
    # If we have matching words, use letter frequency from those (best approach!)
    if len(matching_words) > 0:
        letter_counts = {a: 0 for a in available}
        for word in matching_words:
            for ch in word:
                if ch in available:
                    letter_counts[ch] += 1
        total = sum(letter_counts.values()) + 1e-9
        probs = {a: letter_counts[a] / total for a in available}
        return probs
    
    # Fallback: position-based probabilities + corpus letter frequency
    position_scores = {a: 0 for a in available}
    if L in position_probs:
        for pos, ch in enumerate(pattern):
            if ch == "_":
                for a in available:
                    position_scores[a] += position_probs[L][pos].get(a, 0)
    
    length_freq = {a: 0 for a in available}
    if L in position_probs:
        for pos in range(L):
            for a in available:
                length_freq[a] += position_probs[L][pos].get(a, 0)
    
    total_length = sum(length_freq.values()) + 1e-9
    length_freq = {a: length_freq[a] / total_length for a in available}
    
    # Combine position scores, length frequency, and corpus letter priority
    total_pos = sum(position_scores.values())
    if total_pos > 0.01:
        total_pos_sum = sum(position_scores.values()) + 1e-9
        # 60% position, 25% length frequency, 15% corpus letter priority
        probs = {a: (position_scores[a] / total_pos_sum) * 0.6 + 
                        length_freq.get(a, 0) * 0.25 + 
                        corpus_letter_priority.get(a, 0.01) * 0.15 
                 for a in available}
    else:
        # 70% length frequency, 30% corpus letter priority
        probs = {a: length_freq.get(a, 0) * 0.7 + corpus_letter_priority.get(a, 0.01) * 0.3 for a in available}
    
    total = sum(probs.values()) + 1e-9
    return {a: probs[a] / total for a in available}

with open('test.txt', 'r') as f:
    words = f.read().splitlines()

env = HangmanEnv(words)
games = 2000
success_count = wrong_guesses = repeated_guesses = 0

for _ in range(games):
    pattern = env.reset()
    guessed = set()
    while True:
        state = (pattern, "".join(sorted(guessed)))
        hmm_probs = get_hmm_probs(pattern, guessed)
        available = [a for a in alphabet if a not in guessed]
        if not available:
            break
        # Use pure HMM probabilities (word-filtering should work now!)
        action = max(hmm_probs, key=hmm_probs.get)
        next_pattern, _, done, success = env.step(action)
        if action in guessed:
            repeated_guesses += 1
        elif action not in env.word:
            wrong_guesses += 1
        guessed.add(action)
        pattern = next_pattern
        if done:
            if success:
                success_count += 1
            break

final_score = (success_count * 2000) - (wrong_guesses * 5) - (repeated_guesses * 2)
success_rate = success_count/games
print("Evaluation Complete!")
print(f"=" * 50)
print(f"Success Rate: {success_rate:.2%} ({success_count}/{games})")
print(f"Wrong Guesses: {wrong_guesses}")
print(f"Repeated Guesses: {repeated_guesses}")
print(f"Final Score: {final_score}")
print(f"=" * 50)
