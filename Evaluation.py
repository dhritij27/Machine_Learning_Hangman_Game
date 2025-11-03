import pickle
import random
import numpy as np

with open('models/hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
with open('models/rl_agent.pkl', 'rb') as f:
    rl_agent = pickle.load(f)

position_probs = hmm_model['position_probs']
alphabet = list("abcdefghijklmnopqrstuvwxyz")

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
    probs = {a: 0 for a in available}
    if L in position_probs:
        for pos, ch in enumerate(pattern):
            if ch == "_":
                for a in available:
                    probs[a] += position_probs[L][pos].get(a, 0)
    total = sum(probs.values()) + 1e-9
    return {a: probs[a]/total for a in available}

with open('/content/test.txt', 'r') as f:
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
        scores = {a: rl_agent.Q.get((state,a),0) + 5*hmm_probs.get(a,0) for a in available}
        action = max(scores, key=scores.get)
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
print("Evaluation Complete!")
print(f"Success Rate: {success_count/games:.2%}")
print(f"Wrong Guesses: {wrong_guesses}")
print(f"Repeated Guesses: {repeated_guesses}")
print(f"Final Score: {final_score}")
