import torch.nn as nn


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