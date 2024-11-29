import gym
import numpy as np
import pickle

env = gym.make("MountainCar-v0", render_mode="human")



num_bins = (20, 20) 
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bins = [np.linspace(b[0], b[1], num_bins[i] - 1) for i, b in enumerate(state_bounds)]


with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def discretize_state(state):
    """Convert continuous state to discrete indices."""
    return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))


state, _ = env.reset()
state = discretize_state(state)
for i in range (200):
    terminated = False
    while not terminated:
        action = np.argmax(q_table[state]) 
        next_state, _, terminated, truncated, _ = env.step(action)
        env.render()
        state = discretize_state(next_state)
    env.reset()

env.close()
