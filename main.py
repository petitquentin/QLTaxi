import os
import numpy as np

from src.environment import make_env, make_env_human
from src.train import train, retrain
from src.evaluate import evaluate, evaluate_agent, success_rate
from src.utils import plot_rewards, save_q_table, load_q_table
from src.agent import QLearningAgent

from config import params

def main():
    env = make_env()
    env_test = make_env_human()

    # --- Check if a Q-table already exists ---
    if os.path.exists(params.MODEL_PATH):
        print("[INFO] Loading existing Q-table...")
        q_table = load_q_table(params.MODEL_PATH)

        # Recreate agent with loaded Q-table
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        agent = QLearningAgent(n_states, n_actions)
        agent.q_table = q_table
        agent, rewards = retrain(env, agent)

    else:
        print("[INFO] No saved Q-table found. Training a new agent...")
        agent, rewards = train(env)

    save_q_table(agent, params.MODEL_PATH)
    plot_rewards(rewards)
    print("[INFO] Training complete. Q-table saved.")

    # --- Evaluate agent ---
    print("[INFO] Starting evaluation...")
    evaluate_agent(env, agent)
    success_rate(env, agent)
    evaluate(env_test, agent)

if __name__ == "__main__":
    main()
