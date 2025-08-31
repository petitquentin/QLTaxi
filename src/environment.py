import gymnasium as gym

def make_env():
    return gym.make("Taxi-v3", render_mode="ansi")


def make_env_human():
    return gym.make("Taxi-v3", render_mode="human")
