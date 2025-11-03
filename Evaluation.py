import pickle
import random

with open('../models/hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
with open('../models/rl_agent.pkl', 'rb') as f:
    rl_agent = pickle.load(f)

transition_probs = hmm_model['transition_probs']
emission_probs = hmm_model['emission_probs']
alphabet = list("abcdefghijklmnopqrstuvwxyz")


class HangmanEnv:
    def __init__(self, words, max_lives=6):
        self.words = words
        self.max_lives = max_lives

    def reset(self):
        self.word = random.choice(self.words)
        self.guessed = set()
        self.lives = self.max_lives
        self.pattern = "_" * len(self.word)
        return self.pattern

    def step(self, letter):
        reward = 0
        done = False

        if letter in self.guessed:
            reward -= 2
        elif letter in self.word:
            self.guessed.add(letter)
            new_pattern = list(self.pattern)
            for i, ch in enumerate(self.word):
                if ch == letter:
                    new_pattern[i] = letter
            self.pattern = "".join(new_pattern)
        else:
            self.lives -= 1
            self.guessed.add(letter)

        if "_" not in self.pattern:
            done = True
            success = True
        elif self.lives <= 0:
            done = True
            success = False
        else:
            success = None

        return self.pattern, reward, done, success


with open('../data/corpus.txt', 'r') as f:
    words = f.read().splitlines()

env = HangmanEnv(words)
success_count, wrong_guesses, repeated_guesses = 0, 0, 0

games = 2000
for g in range(games):
    pattern = env.reset()
    guessed = set()
    while True:
        state = (pattern, "".join(sorted(guessed)))
        available = [a for a in alphabet if a not in guessed]
        hmm_probs = {a: emission_probs.get(a, 0) for a in available}
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

print(f"Evaluation Complete!")
print(f"Success Rate: {success_count/games:.2%}")
print(f"Wrong Guesses: {wrong_guesses}")
print(f"Repeated Guesses: {repeated_guesses}")
print(f"Final Score: {final_score}")
