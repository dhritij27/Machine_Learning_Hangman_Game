import pickle
import random
import numpy as np
from collections import defaultdict

with open('models/hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)

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
        reward = 0
        done = False
        if letter in self.guessed:
            reward -= 4
        elif letter in self.word:
            self.guessed.add(letter)
            new_pattern = list(self.pattern)
            for i, ch in enumerate(self.word):
                if ch == letter:
                    new_pattern[i] = letter
            diff = new_pattern.count(letter) - self.pattern.count(letter)
            self.pattern = "".join(new_pattern)
            reward += 12 + 4 * diff
        else:
            self.lives -= 1
            self.guessed.add(letter)
            reward -= 10
        if "_" not in self.pattern:
            reward += 150
            done = True
        elif self.lives <= 0:
            reward -= 100
            done = True
        return self.pattern, reward, done

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

with open('/content/corpus.txt', 'r') as f:
    words = f.read().splitlines()

env = HangmanEnv(words)
agent = QLearningAgent()
episodes = 12000
scores = []
wins = 0

for ep in range(episodes):
    pattern = env.reset()
    guessed = set()
    total_reward = 0
    while True:
        state = agent.get_state(pattern, guessed)
        action = agent.choose_action(state, pattern, guessed)
        next_pattern, reward, done = env.step(action)
        next_state = agent.get_state(next_pattern, env.guessed)
        agent.update(state, action, reward, next_state)
        total_reward += reward
        guessed.add(action)
        pattern = next_pattern
        if done:
            if "_" not in pattern:
                wins += 1
            break
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    scores.append(total_reward)
    if ep % 500 == 0 and ep > 0:
        rate = wins/ep
        avg = np.mean(scores[-500:])
        print(f"Episode {ep:5d} | WinRate {rate:6.2%} | AvgReward {avg:8.2f} | Eps {agent.epsilon:6.3f}")

with open('models/rl_agent.pkl','wb') as f:
    pickle.dump(agent,f)

print("RL agent trained and saved.")
