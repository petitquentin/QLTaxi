import numpy as np

def evaluate_agent(env, agent, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.argmax(agent.q_table[state])  # always exploit
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)

    avg_reward = sum(total_rewards) / episodes
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward

def success_rate(env, agent, episodes=100):
    successes = 0
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        while not done:
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward == 20:  # +20 reward = successful drop
                successes += 1
                break
    rate = successes / episodes
    print(f"Success rate: {rate*100:.2f}%")
    return rate

def evaluate(env, agent, episodes=5):
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        env.render()
        while not done:
            action = agent.choose_action(state)
            state, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()
