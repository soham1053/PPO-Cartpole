import numpy as np
import torch
import torch.optim as optim
from utils import probsToIndex

class AdvantageActorCriticAgent:
    def __init__(self, input_size, output_size, policy_net, value_net,
                 batch_size=512, gamma=1.0,
                 name="a2c"):
        self.input_size = input_size
        self.output_size = output_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy = policy_net(input_size, output_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters())
        # Make final layer's weights tiny and biases 0 for maximum entropy --> exploration
        for name, param in self.policy.named_parameters():
            if name == "fcfinal.weight":
                param.data.normal_(0, 0.001)
            elif name == "fcfinal.bias":
                param.data.fill_(0)

        self.value = value_net(input_size).to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters())

        self.gamma = gamma
        self.batch_size = batch_size
        self.timestep = 0

        self.reset()

        self.name = name

    def reset(self):
        self.policy_loss = torch.zeros([1]).to(self.device)
        self.action_prob_dists = torch.zeros(self.output_size).to(self.device)
        self.stack_dists = False
        self.states = [[]]
        self.actions = []
        self.rewards = [[]]

    def take_action(self, state):
        self.states[-1].append(torch.from_numpy(state.copy().astype(np.float32)))
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
        self.timestep += 1

        self.rewards[-1].append(reward)

        if self.timestep % self.batch_size == 0:
            td_errors = []
            value_losses = []
            for episode in range(len(self.states)):
                for t in range(len(self.states[episode])):
                    self.value_optimizer.zero_grad()
                    state = self.states[episode][t].to(self.device)
                    reward = self.rewards[episode][t]
                    if episode == len(self.states) - 1 and t == len(self.states[episode]) - 1 and not done:
                        continue
                    elif t < len(self.states[episode]) - 1:
                        nextState = self.states[episode][t + 1].to(self.device)
                        td_error = reward + self.gamma * self.value(nextState) - self.value(state)
                    else:
                        td_error = reward - self.value(state)
                    td_errors.append(td_error)

                    value_loss = td_error ** 2
                    value_losses.append(value_loss.item())
                    value_loss.backward()
                    self.value_optimizer.step()

            self.policy_optimizer.zero_grad()
            for i in range(len(td_errors)):
                self.policy_loss -= torch.log(self.action_prob_dists[i][self.actions[i]]) * td_errors[i].item()

            self.policy_loss /= len(td_errors)
            policy_loss_item = self.policy_loss.item()
            self.policy_loss.backward()
            self.policy_optimizer.step()

            self.reset()

            lossOut = {"policyLoss": [policy_loss_item], "valueLosses": value_losses}
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