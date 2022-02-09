import matplotlib.pyplot as plt
import torch
import numpy as np


def plotReturns(values, moving_avg_period):
    plt.figure(100)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(values, 'blue')

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg, 'green')
    plt.pause(0.001)


def plotLosses(values, moving_avg_period):
    plt.figure(101)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(values, 'red')

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg, 'orange')
    plt.pause(0.001)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
    else:
        moving_avg = torch.zeros(len(values))
    return moving_avg.numpy()


# From softmax probabilities, choose index of action
def probsToIndex(probs_tensor):
    probs_np = probs_tensor.detach().cpu().numpy()
    index = np.random.choice(probs_np.size, p=probs_np / probs_np.sum())
    return index