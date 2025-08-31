import numpy as np
from src.agent import QLearningAgent
from config import params

def train(env, episodes=params.EPISODES):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = QLearningAgent(n_states, n_actions)

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Décroissance epsilon
        if agent.epsilon > params.EPSILON_MIN:
            agent.epsilon *= params.EPSILON_DECAY

    return agent, rewards_per_episode


def retrain(env, agent, episodes=params.EPISODES):
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Décroissance epsilon
        if agent.epsilon > params.EPSILON_MIN:
            agent.epsilon *= params.EPSILON_DECAY

    return agent, rewards_per_episode
