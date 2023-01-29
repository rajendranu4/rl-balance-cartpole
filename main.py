import gym

from QLearning import *
from GradientPolicy import *


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    print("Q Learning for cartpole environment")
    q_learn_cp = QLearningCartpole(env)
    q_learn_cp.init_qtable()
    q_learn_cp.agent_train(n_episodes=10000)

    # print("\n\nPolicy Gradient for cartpole environment")
    # agentgp = GradientPolicyCartpole(env)
    # agentgp.agent_train(n_episodes=5000)
    # agentgp.test_n_episodes(n_episodes=100)
    # agentgp.test_one_episode()