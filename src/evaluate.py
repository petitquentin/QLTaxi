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
