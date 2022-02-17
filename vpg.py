import numpy as np
import torch
import torch.optim as optim
import itertools
from utils import probsToIndex

class VanillaPolicyGradientAgent:
    def __init__(self, input_size, output_size, policy_net, batch_size=1, gamma=1.0):
        self.input_size = input_size
        self.output_size = output_size

        self.policy = policy_net(input_size, output_size)
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
        self.timestep = 0

        self.reset()

    def reset(self):
        self.optimizer.zero_grad()
        self.loss = torch.zeros([]).to(self.device)
        self.action_prob_dists = torch.zeros(self.output_size).to(self.device)
        self.stack_dists = False
        self.actions = []
        self.advantages = []

    def take_action(self, state):
        state = torch.from_numpy(state.astype(np.float32)).to(self.device)
        action_probs = self.policy(state)
        action = probsToIndex(action_probs)

        if self.stack_dists:
            self.action_prob_dists = torch.vstack((self.action_prob_dists, action_probs))
        else:
            self.action_prob_dists = action_probs
            self.stack_dists = True

        self.actions.append(action)

        return action

    def learn(self, rewards):
        self.timestep += 1

        self.advantages.append([])
        for r in rewards:
            self.advantages[-1].append(0)
            for t in range(len(self.advantages[-1]) - 1, -1, -1):
                self.advantages[-1][t] += r * (self.gamma ** (len(self.advantages[-1]) - 1 - t))

        if self.timestep % self.batch_size == 0:
            advantages_flat = list(itertools.chain(*self.advantages))
            for t in range(len(advantages_flat)):
                self.loss -= advantages_flat[t] * self.action_prob_dists[t][self.actions[t]]
            self.loss /= len(advantages_flat)

            loss_value = self.loss.item()

            self.loss.backward()
            self.optimizer.step()

            self.reset()

            return {"hasLoss": True, "loss": loss_value}
        else:
            return {"hasLoss": False}

    def save(self, path):
        torch.save(self.policy.state_dict(), path)