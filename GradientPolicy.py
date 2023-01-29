import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical, Bernoulli

np.random.seed(0)


class GradientPolicyNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # feed forward network with 2 hidden layers, input layer with 4 units and output layer with 2 units
        # ReLU activation function for the hidden layers
        # Softmax funtion for the output layer since output is probability distribution of 2 variables
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.hidden_neurons = 10

        self.hidden_layer1 = nn.Linear(4, self.hidden_neurons)
        self.hidden_layer2 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.output_layer = nn.Linear(self.hidden_neurons, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = self.hidden_layer1(state)
        x = self.relu(x)
        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)

        return x


class GradientPolicyCartpole:
    def __init__(self, env):
        self.env = env
        self.agent = GradientPolicyNetwork(self.env)
        self.optimizer = torch.optim.Adam(self.agent.parameters())

        # to store the states, actions and rewards of a complete episode during training
        self.rewards = []
        self.actions = []
        self.states = []

    def rewards_sum(self):
        return torch.tensor(np.sum(self.rewards))

    def clean_path(self):
        self.rewards = []
        self.actions = []
        self.states = []

    def update_path(self, reward, action, state):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)

    def episode_path(self):

        self.states = torch.FloatTensor(self.states)
        self.actions = torch.FloatTensor(self.actions)

        distributions = self.agent(self.states)

        # loss function --> negative log probabilities of action * reward --> gradient ascent
        # gradient ascent - to maximize the rewards for the parameters weights in the network
        loss = torch.sum(-Categorical(distributions).log_prob(self.actions) * self.rewards_sum())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def agent_train(self, n_episodes=10000):
        episode = 0
        max_mean_reward = 0
        max_total_reward = 0
        max_reward_count = 0

        while max_mean_reward < 200 and episode < n_episodes:
            episode += 1

            self.clean_path()

            state = self.env.reset()
            done = False
            rewards_episode = 0
            while not done:
                # get the probability distribution of actions with the given state
                action_dist = self.agent(torch.FloatTensor(state).unsqueeze(0))

                # sampling the action probability distributions to pick action
                # probability of action on [0.6, 0.4] distribution is 60% chance for 1 and 40% chance for 0
                action = Categorical(action_dist).sample()

                # action from current state is action.item() since action is a torch tensor
                next_state, reward, done, info = self.env.step(action.item())

                rewards_episode += reward

                self.update_path(reward, action, state)

                state = next_state

            if rewards_episode >= 200:
                max_reward_count += 1

            max_total_reward += rewards_episode

            if episode % 20 == 0:
                max_mean_reward = max_total_reward / 20
                max_total_reward = 0

            if episode % 1000 == 0 or rewards_episode > 195:
                print("Episode: {} ==> Reward: {}".format(episode, rewards_episode))

                # running the neural network with the recent episode's states to update the parameters
            self.episode_path()

        print("\nPolicy Gradient - Training ends at episode {}".format(episode))
        print("Total number of Maximum rewards (200 timesteps) achieved during training: " + str(max_reward_count))

    def test_n_episodes(self, n_episodes):
        print("\nTesting {} episodes of cartpole agent".format(n_episodes))
        total_reward = 0
        max_rewards_count = 0
        for k in range(n_episodes):
            state = self.env.reset()
            done = False
            rewards = 0

            while not done:
                # env.render()
                action_dist = self.agent(torch.FloatTensor(state).unsqueeze(0))
                action = Categorical(action_dist).sample()

                next_state, reward, done, info = self.env.step(action.item())

                rewards += reward

                state = next_state

            if rewards >= 200:
                max_rewards_count += 1

            total_reward += rewards

        print("Mean rewards for 100 episodes: {}".format(total_reward / 100))
        print("Number of episodes with 200 rewards: {}".format(max_rewards_count))

    def test_one_episode(self):
        print("\nTesting One episode of cartpole agent")
        state = self.env.reset()
        done = False
        rewards = 0

        while not done:
            self.env.render()
            action_dist = self.agent(torch.FloatTensor(state).unsqueeze(0))
            action = Categorical(action_dist).sample()

            next_state, reward, done, info = self.env.step(action.item())

            rewards += reward

            state = next_state

        print("Total rewards for this episode: " + str(rewards))
        print("Info: {}".format(info))

        self.env.close()