import gym
from train import Policy
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from utils import probsToIndex

env = gym.make("CartPole-v1")

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
policy = Policy(input_size, output_size)
policy.load_state_dict(torch.load("policy.pt"))
policy.eval()

returns = []
for i in range(10):
    state = env.reset()
    returns.append(0)
    done = False
    while not done:
        state = torch.from_numpy(state.astype(np.float32))
        action_probs = policy(state)
        action = probsToIndex(action_probs)
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.01)

        returns[-1] += reward
env.close()

plt.hist(returns)
plt.show()