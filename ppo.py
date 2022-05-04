import numpy as np
import torch
import torch.optim as optim
import itertools
from utils import probsToIndex

class ProximalPolicyOptimizationAgent:
    def __init__(self, input_size, output_size, policy_net, value_net,
                 gamma=1.0, alpha=0.0002, epsilon=0.2, gae_lambda=0.95, batch_size=20, minibatch_size=5, n_epochs=4,
                 name="ppo"):
        self.input_size = input_size
        self.output_size = output_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy = policy_net(input_size, output_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        # Make final layer's weights tiny and biases 0 for maximum entropy --> exploration
        for nm, param in self.policy.named_parameters():
            if nm == "fcfinal.weight":
                param.data.normal_(0, 0.001)
            elif nm == "fcfinal.bias":
                param.data.fill_(0)

        self.value = value_net(input_size).to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=alpha)

        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.timestep = 0

        self.reset()

        self.name = name

    def reset(self):
        self.action_prob_dists = torch.zeros(self.output_size).to(self.device)
        self.stack_dists = False
        self.states = [[]]
        self.actions = []
        self.rewards = [[]]

    def take_action(self, state):
        self.states[-1].append(torch.from_numpy(state.copy().astype(np.float32)))
        state = torch.from_numpy(state.astype(np.float32)).to(self.device)
        action_prob_dist = self.policy(state).detach()
        action = probsToIndex(action_prob_dist)

        if self.stack_dists:
            self.action_prob_dists = torch.vstack((self.action_prob_dists, action_prob_dist))
        else:
            self.action_prob_dists = action_prob_dist
            self.stack_dists = True

        self.actions.append(action)

        return action

    def learn(self, reward, done):
        self.timestep += 1

        self.rewards[-1].append(reward)

        if self.timestep % self.batch_size == 0:
            values = []
            for episode in range(len(self.states)):
                values.append([])
                for t in range(len(self.states[episode])):
                    state = self.states[episode][t].to(self.device)
                    values[-1].append(self.value(state))

            advantages = []
            for episode in range(len(self.states)):
                for t in range(len(self.states[episode])):
                    discount = 1
                    advantage = 0
                    for i in range(t, len(self.states[episode])):
                        reward = self.rewards[episode][i]
                        if episode == len(self.states) - 1 and i == len(self.states[episode]) - 1 and not done:
                            pass
                        elif i < len(self.states[episode]) - 1:
                            advantage += discount * (reward + self.gamma * values[episode][i + 1] - values[episode][i])
                        else:
                            advantage += discount * (reward - values[episode][i])
                        discount *= self.gamma * self.gae_lambda
                    advantages.append(advantage)

            actions = torch.tensor(self.actions)
            states_flat = torch.stack(list(itertools.chain(*self.states))).to(self.device)
            values = torch.stack(list(itertools.chain(*values))).to(self.device).detach()
            advantages = torch.tensor(advantages).to(self.device)

            policy_losses = []
            value_losses = []
            for epoch in range(self.n_epochs):
                permutation = torch.randperm(self.batch_size)

                for i in range(0, self.batch_size, self.minibatch_size):
                    indices = permutation[i : i + self.minibatch_size]

                    old_action_probs = self.action_prob_dists[indices, actions[indices]]
                    action_probs = self.policy(states_flat[indices])[:, actions[indices]]

                    prob_ratios = action_probs / old_action_probs

                    weighted_prob_ratios = prob_ratios * advantages[indices]
                    weighted_clipped_prob_ratios = torch.clamp(prob_ratios, 1-self.epsilon, 1+self.epsilon) * advantages[indices]

                    policy_loss = (-torch.min(weighted_prob_ratios, weighted_clipped_prob_ratios)).mean()
                    policy_losses.append(policy_loss.item())

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    value_loss = ((advantages[indices] + values[indices] - self.value(states_flat[indices])) ** 2).mean()
                    value_losses.append(value_loss.item())

                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    self.value_optimizer.step()

            self.reset()

            lossOut = {"policyLoss": policy_losses, "valueLosses": value_losses}
        else:
            lossOut = {}

        if done:
            self.states.append([])
            self.rewards.append([])

        return lossOut

    def save(self):
        torch.save(self.policy.state_dict(), self.name + "Policy.pt")
        torch.save(self.value.state_dict(), self.name + "Value.pt")

    def load(self):
        self.policy.load_state_dict(torch.load(self.name + "Policy.pt"))
        self.value.load_state_dict(torch.load(self.name + "Value.pt"))