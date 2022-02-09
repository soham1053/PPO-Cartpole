import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utils import plotReturns, plotLosses, probsToIndex


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fcfinal = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fcfinal(x)
        return self.softmax(x)


def run():
    env = gym.make("CartPole-v1")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = Policy(input_size, output_size).to(device)
    optimizer = optim.Adam(policy.parameters())

    # Make final layer's weights tiny and biases 0 for maximum entropy --> exploration
    for name, param in policy.named_parameters():
        if name == "fcfinal.weight":
            param.data.normal_(0, 0.001)
        elif name == "fcfinal.bias":
            param.data.fill_(0)

    # todo hyperparameter tuning
    gamma = 1.0

    returns = []
    losses = []
    log_period = 100
    # todo make sure everything is correct-o
    # todo make big batches of multiple rollouts
    for batch_idx in range(100000):
        optimizer.zero_grad()

        state = env.reset()

        loss = torch.zeros([]).to(device)
        action_prob_dists = torch.zeros(output_size).to(device)
        actions = []
        advantages = []

        returns.append(0)

        timestep = 0
        done = False
        while not done:
            state = torch.from_numpy(state.astype(np.float32)).to(device)
            action_probs = policy(state)
            action = probsToIndex(action_probs)
            state, reward, done, _ = env.step(action)

            # todo implement a memory class that updates all of this
            if timestep == 0:
                action_prob_dists = action_probs
            else:
                action_prob_dists = torch.vstack((action_prob_dists, action_probs))
            actions.append(action)
            advantages.append(0)
            for t in range(len(advantages) - 1, -1, -1):
                advantages[t] += reward * (gamma**(len(advantages) - 1 - t))

            returns[-1] += reward

            timestep += 1

        for t in range(timestep):
            loss -= advantages[t] * action_prob_dists[t][actions[t]]
        loss /= timestep

        losses.append(loss.item())

        if batch_idx % log_period == 0:
            plotReturns(returns, 100)
            plotLosses(losses, 100)
            # torch.save(policy.state_dict(), 'policy.pt')

        loss.backward()
        optimizer.step()

    env.close()
    plt.show()


if __name__ == '__main__':
    run()