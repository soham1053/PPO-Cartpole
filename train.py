import gym
import matplotlib.pyplot as plt
from utils import plotReturns, plotLosses
from nets import Policy, Value
from vpg import VanillaPolicyGradientAgent
from a2c import AdvantageActorCriticAgent
from ppo import ProximalPolicyOptimizationAgent


def run():
    env = gym.make("CartPole-v1")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    agent = ProximalPolicyOptimizationAgent(input_size, output_size, Policy, Value, name="ppo")

    returns = []
    policyLosses = []
    valueLosses = []
    log_period = 5
    save_period = 100
    for episode in range(1, 50000):
        state = env.reset()

        rewards = []
        returns.append(0)

        done = False
        while not done:
            action = agent.take_action(state)
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            returns[-1] += reward

            loss = agent.learn(reward, done)
            if "policyLoss" in loss:
                policyLosses.append(sum(loss["policyLoss"]) / len(loss["policyLoss"]))
            if "valueLosses" in loss:
                valueLosses.append(sum(loss["valueLosses"]) / len(loss["valueLosses"]))

        if episode % log_period == 0:
            plotReturns(returns, 100, idx=1, color='blue')
            if len(policyLosses) > 0:
                plotLosses(policyLosses, 100, "Policy", idx=2, color='red')
            if len(valueLosses) > 0:
                plotLosses(valueLosses, 100, "Value", idx=3, color='purple')
        if episode % save_period == 0:
            agent.save()

    env.close()
    plt.show()


if __name__ == '__main__':
    run()