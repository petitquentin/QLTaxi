import os
import numpy as np
import matplotlib.pyplot as plt

from config import params


def save_q_table(agent, path=params.MODEL_PATH):
    """Save the Q-table as a NumPy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, agent.q_table)

def load_q_table(path=params.MODEL_PATH):
    """Load the Q-table from a NumPy file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No Q-table found at {path}")
    return np.load(path)

def plot_rewards(rewards, save_path=os.path.join(params.RESULTS_DIR, "rewards.png")):
    """Plot and save the rewards evolution during training."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.title("Training progress (Q-learning on Taxi-v3)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Reward plot saved at {save_path}")