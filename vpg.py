import numpy as np
import torch
import torch.optim as optim
import itertools
from utils import probsToIndex

class VanillaPolicyGradientAgent:
    def __init__(self, input_size, output_size, policy_net,
                 batch_size=5, gamma=1.0,
                 name="vpg"):
        self.input_size = input_size
        self.output_size = output_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy_net(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters())
        # Make final layer's weights tiny and biases 0 for maximum entropy --> exploration
        for name, param in self.policy.named_parameters():
            if name == "fcfinal.weight":
                param.data.normal_(0, 0.001)
            elif name == "fcfinal.bias":
                param.data.fill_(0)

        self.gamma = gamma
        self.batch_size = batch_size
        self.episode = 0

        self.reset()

        self.name = name

    def reset(self):
        self.optimizer.zero_grad()
        self.loss = torch.zeros([]).to(self.device)
        self.action_prob_dists = torch.zeros(self.output_size).to(self.device)
        self.stack_dists = False
        self.actions = []
        self.episodeRewards = []
        self.advantages = []

    def take_action(self, state):
        state = torch.from_numpy(state.astype(np.float32)).to(self.device)
        action_prob_dist = self.policy(state)
        action = probsToIndex(action_prob_dist)

        if self.stack_dists:
            self.action_prob_dists = torch.vstack((self.action_prob_dists, action_prob_dist))
        else:
            self.action_prob_dists = action_prob_dist
            self.stack_dists = True

        self.actions.append(action)

        return action

    def learn(self, reward, done):
        if not done:
            self.episodeRewards.append(reward)
            return {"hasLoss": False}

        self.episode += 1

        self.advantages.append([])
        for r in self.episodeRewards:
            self.advantages[-1].append(0)
            for t in range(len(self.advantages[-1]) - 1, -1, -1):
                self.advantages[-1][t] += r * (self.gamma ** (len(self.advantages[-1]) - 1 - t))
        self.episodeRewards.clear()

        if self.episode % self.batch_size == 0:
            advantages_flat = list(itertools.chain(*self.advantages))
            for t in range(len(advantages_flat)):
                self.loss -= advantages_flat[t] * torch.log(self.action_prob_dists[t][self.actions[t]])
            self.loss /= len(advantages_flat)

            loss_value = self.loss.item()

            self.loss.backward()
            self.optimizer.step()

            self.reset()

            return {"policyLoss": [loss_value]}
        else:
            return {}

    def save(self):
        torch.save(self.policy.state_dict(), self.name + "Policy.pt")

    def load(self):
        self.policy.load_state_dict(torch.load(self.name + "Policy.pt"))