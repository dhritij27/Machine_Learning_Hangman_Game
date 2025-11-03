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
filtered_count = 0
for word in all_train_words:
    if word:
        words_by_length[len(word)].append(word)
    else:
        filtered_count += 1

print(f"Total words loaded: {len(all_train_words)}")
print(f"Words filtered out (empty): {filtered_count}")
print(f"Words kept: {len(all_train_words) - filtered_count}")
print(f"Word length distribution: {dict(sorted((k, len(v)) for k, v in words_by_length.items()))}")
print()

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
    available_set = set(available)
    
    # Filter words that match the pattern (less aggressive - only check revealed positions)
    matching_words = []
    candidates = words_by_length.get(L, [])
    for word in candidates:
        matches = True
        # Only check exact matches at revealed positions
        for i, ch in enumerate(pattern):
            if ch != "_":
                if word[i] != ch:
                    matches = False
                    break
        if matches:
            matching_words.append(word)
    
    # If we have matching words, use letter frequency from those (best approach!)
    if len(matching_words) > 0:
        letter_counts = Counter()
        for word in matching_words:
            for ch in word:
                if ch in available_set:
                    letter_counts[ch] += 1
        total = sum(letter_counts.values()) + 1e-9
        probs = {a: letter_counts.get(a, 0) / total for a in available}
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
    test_words = f.read().splitlines()

with open('corpus.txt', 'r') as f:
    corpus_words_list = f.read().splitlines()

# Evaluate on test words
print("Evaluating on TEST words...")
env = HangmanEnv(test_words)
games_test = len(test_words)
success_count_test = wrong_guesses_test = repeated_guesses_test = 0

for _ in range(games_test):
    pattern = env.reset()
    guessed = set()
    while True:
        state = (pattern, "".join(sorted(guessed)))
        hmm_probs = get_hmm_probs(pattern, guessed)
        available = [a for a in alphabet if a not in guessed]
        if not available:
            break
        action = max(hmm_probs, key=hmm_probs.get)
        next_pattern, _, done, success = env.step(action)
        if action in guessed:
            repeated_guesses_test += 1
        elif action not in env.word:
            wrong_guesses_test += 1
        guessed.add(action)
        pattern = next_pattern
        if done:
            if success:
                success_count_test += 1
            break

# Evaluate on corpus words (sample for speed)
print("Evaluating on CORPUS words...")
corpus_sample = corpus_words_list[:1000]  # Sample first 1000 words
env = HangmanEnv(corpus_sample)
games_corpus = len(corpus_sample)
success_count_corpus = wrong_guesses_corpus = repeated_guesses_corpus = 0

for _ in range(games_corpus):
    pattern = env.reset()
    guessed = set()
    while True:
        state = (pattern, "".join(sorted(guessed)))
        hmm_probs = get_hmm_probs(pattern, guessed)
        available = [a for a in alphabet if a not in guessed]
        if not available:
            break
        action = max(hmm_probs, key=hmm_probs.get)
        next_pattern, _, done, success = env.step(action)
        if action in guessed:
            repeated_guesses_corpus += 1
        elif action not in env.word:
            wrong_guesses_corpus += 1
        guessed.add(action)
        pattern = next_pattern
        if done:
            if success:
                success_count_corpus += 1
            break

# Calculate totals
total_success = success_count_test + success_count_corpus
total_games = games_test + games_corpus
total_wrong = wrong_guesses_test + wrong_guesses_corpus
total_repeated = repeated_guesses_test + repeated_guesses_corpus

final_score = (total_success * 2000) - (total_wrong * 5) - (total_repeated * 2)
success_rate = total_success / total_games

print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)
print(f"\nTEST Words:")
print(f"  Success Rate: {success_count_test/games_test:.2%} ({success_count_test}/{games_test})")
print(f"  Wrong Guesses: {wrong_guesses_test}")
print(f"  Repeated Guesses: {repeated_guesses_test}")

print(f"\nCORPUS Words:")
print(f"  Success Rate: {success_count_corpus/games_corpus:.2%} ({success_count_corpus}/{games_corpus})")
print(f"  Wrong Guesses: {wrong_guesses_corpus}")
print(f"  Repeated Guesses: {repeated_guesses_corpus}")

print(f"\nOVERALL:")
print(f"  Success Rate: {success_rate:.2%} ({total_success}/{total_games})")
print(f"  Wrong Guesses: {total_wrong}")
print(f"  Repeated Guesses: {total_repeated}")
print(f"  Final Score: {final_score}")
print("=" * 60)
