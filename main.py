import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)


def probsToAction(action_probs):
    action_probs_np = action_probs.detach().cpu().numpy()
    action = np.random.choice(action_probs_np.size, p=action_probs_np / action_probs_np.sum())
    return action

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)

policy = Policy(4, 2).to(device)
optimizer = optim.Adam(policy.parameters())

env = gym.make("CartPole-v1")

returns = []
for i in range(50000):
    optimizer.zero_grad()

    state = env.reset()
    returns.append(0)
    loss = torch.zeros([]).to(device)
    done = False
    while not done:
        state = torch.from_numpy(state.astype(np.float32)).to(device)
        action_probs = policy(state)
        action = probsToAction(action_probs)
        state, reward, done, _ = env.step(action)
        loss += -reward * torch.log(action_probs[action])
        returns[i] += reward

        if done:
            plot(returns, 100)

    loss.backward()
    optimizer.step()

# returns = []
# for i in range(10000):
#     optimizer.zero_grad()
#
#     state = env.reset()
#     rewards = []
#     done = False
#     while not done:
#         state = torch.from_numpy(state.astype(np.float32)).to(device)
#         action_probs = policy(state)
#         action = probsToAction(action_probs)
#         state, reward, done, _ = env.step(action)
#         rewards.insert(reward, 0)
#
#     returns.append(sum(rewards))
#
# env.close()