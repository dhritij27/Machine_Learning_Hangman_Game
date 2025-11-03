import pickle
import random
import numpy as np
from collections import defaultdict

with open('../models/hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)

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
            reward += 10
        else:
            self.lives -= 1
            self.guessed.add(letter)
            reward -= 5

        if "_" not in self.pattern:
            reward += 100
            done = True
        elif self.lives <= 0:
            reward -= 100
            done = True

        return self.pattern, reward, done


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_state(self, pattern, guessed):
        return (pattern, "".join(sorted(guessed)))

    def choose_action(self, state, hmm_probs):
        available = [a for a in alphabet if a not in state[1]]
        if random.random() < self.epsilon:
            return random.choice(available)
       
        scores = {a: self.Q[(state, a)] + hmm_probs.get(a, 0) for a in available}
        return max(scores, key=scores.get)

    def update(self, state, action, reward, next_state):
        max_next = max([self.Q[(next_state, a)] for a in alphabet], default=0)
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * max_next - self.Q[(state, action)])


with open('../data/corpus.txt', 'r') as f:
    words = f.read().splitlines()

env = HangmanEnv(words)
agent = QLearningAgent()

episodes = 5000
scores = []

for ep in range(episodes):
    pattern = env.reset()
    guessed = set()
    total_reward = 0

    while True:
        state = agent.get_state(pattern, guessed)
        hmm_probs = {a: emission_probs.get(a, 0) for a in alphabet}
        action = agent.choose_action(state, hmm_probs)
        next_pattern, reward, done = env.step(action)
        next_state = agent.get_state(next_pattern, env.guessed)
        agent.update(state, action, reward, next_state)
        total_reward += reward
        guessed.add(action)
        pattern = next_pattern
        if done:
            break

    agent.epsilon *= agent.epsilon_decay
    scores.append(total_reward)
    if ep % 500 == 0:
        print(f"Episode {ep} | Reward: {total_reward}")


with open('../models/rl_agent.pkl', 'wb') as f:
    pickle.dump(agent, f)

print(" RL Agent trained and saved!")
