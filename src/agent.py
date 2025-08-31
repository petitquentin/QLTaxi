import numpy as np
import random

from config import params

class QLearningAgent:
    def __init__(self, n_states, n_actions):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = params.ALPHA
        self.gamma = params.GAMMA
        self.epsilon = params.EPSILON

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.q_table.shape[1])  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
