import torch.nn as nn
from collections import OrderedDict


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, 256)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(256, 256)),
            ('relu2', nn.ReLU()),
            ('fcfinal', nn.Linear(256, output_size)),
            ('softmax', nn.Softmax(-1))
        ]))
    def forward(self, x):
        return self.model(x)


class Value(nn.Module):
    def __init__(self, input_size):
        super(Value, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, 256)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(256, 256)),
            ('relu2', nn.ReLU()),
            ('fcfinal', nn.Linear(256, 1)),
        ]))

    def forward(self, x):
        return self.model(x)