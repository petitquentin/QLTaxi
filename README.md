# Taxi-v3 Q-Learning Agent

A Python project that implements a Q-learning agent to solve the Taxi-v3 environment from Gymnasium.

The agent learns to pick up and drop off passengers efficiently by maximizing total rewards.

## Project Overview

The goal of this project is to:
- Train a Q-learning agent to navigate a taxi grid, pick up passengers, and drop them at their destination.
- Store and reuse the Q-table for future training or evaluation.
- Evaluate the agent’s performance using average reward and success rate.
- Provide a modular, readable codebase suitable for learning reinforcement learning (RL).

## Features
- Train a new agent or retrain an existing one using a saved Q-table.
- Evaluate agent performance and visualize training rewards.
- Fully compatible with Gymnasium (successor of Gym).
- Modular project structure for easy experimentation.
- Automatic saving of Q-table and reward plots in dedicated folders.

## Project Structure

```
QLTaxi/
│── README.md                # Project description
│── requirements.txt         # Python dependencies
│── main.py                  # Main script to train/evaluate
│
├── config/
│   └── params.py            # Hyperparameters
│
├── src/
│   ├── environment.py       # Gymnasium environment wrapper
│   ├── agent.py             # QLearningAgent class
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation functions
│   └── utils.py             # Save/load Q-table, plot rewards
│
├── models/
│   └── q_table.npy          # Saved Q-table
│
└── results/
    └── rewards.png          # Training reward plots
```

## Requirements

Python 3.9+ with the following packages:
```
gymnasium
numpy
matplotlib
```

Install via pip:
```
pip install gymnasium numpy matplotlib
```

For Taxi-v3 specifically:
```
pip install gymnasium[toy_text]
```

## How to Run

### Train or retrain an agent
```python
python -m src.main
```
- If a saved Q-table exists, it will load and optionally retrain.
- Reward plots are automatically saved in ```results/rewards.png```

### Evaluate a trained agent

- The main.py script automatically evaluates after training/loading.
- The metrics used are:
  - Average Reward: Average total reward per episode.
  - Success Rate: Percentage of episodes where the passenger was successfully dropped at the correct location
