import gym
import matplotlib.pyplot as plt
from utils import plotReturns, plotLosses
from nets import Policy
from vpg import VanillaPolicyGradientAgent


def run():
    env = gym.make("CartPole-v1")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # todo hyperparameter tuning
    agent = VanillaPolicyGradientAgent(input_size, output_size, Policy, batch_size=100, gamma=1.0)

    returns = []
    losses = []
    log_period = 100
    for episode in range(1, 1000001):
        state = env.reset()

        rewards = []
        returns.append(0)

        done = False
        while not done:
            action = agent.take_action(state)
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            returns[-1] += reward

        loss = agent.learn(rewards)
        if loss["hasLoss"]:
            losses.append(loss["loss"])

        if episode % log_period == 0:
            plotReturns(returns, 100)
            plotLosses(losses, 100)
            agent.save('policy.pt')

    env.close()
    plt.show()


if __name__ == '__main__':
    run()