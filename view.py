import gym
from train import Policy, Value
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import probsToIndex

env = gym.make("CartPole-v1")

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
agentName = "ppo"

policy = Policy(input_size, output_size)
policy.load_state_dict(torch.load(agentName + "Policy.pt"))
policy.eval()
value = Value(input_size)
value.load_state_dict(torch.load(agentName + "Value.pt"))
value.eval()

values = []
returns = []
for i in range(10):
    state = env.reset()
    returns.append(0)
    done = False
    while not done:
        state = torch.from_numpy(state.astype(np.float32))
        values.append(value(state).item())
        action_probs = policy(state)
        # print(action_probs)
        action = probsToIndex(action_probs)
        state, reward, done, _ = env.step(action)
        env.render()

        # Plot values
        # plt.figure(0)
        # plt.clf()
        # plt.xlabel("Timestep")
        # plt.ylabel("Value")
        # plt.plot(values)
        # plt.pause(0.0001)

        returns[-1] += reward

env.close()

plt.figure(1)
plt.hist(returns)
plt.show()