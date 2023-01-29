import numpy as np
np.random.seed(0)


class QLearningCartpole:
    def __init__(self, env):
        self.env = env
        # Q-Value formula parameters
        self.lr = 0.01
        self.discount = 0.9

        # epsilon decay function parameters
        # epsilon is utilized to determine if the current action is chosen randomly or from the q-table
        self.epsilon = 1
        self.epsilon_decay = 0.95

        self.discrete_bins = [6, 6, 6, 13]

    def init_qtable(self):
        self.q_table = np.zeros(self.discrete_bins + [self.env.action_space.n, ])

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    def continuous_to_discrete_state(self, cont_state):
        # converting continuous values of state space to discrete values
        # q learning algorithm needs discrete state space to construct q-table

        disc_state = [0, 0, 0, 0]

        # cont_state[0] is between -4.8 to 4.8
        if (cont_state[0] < -4.8):
            disc_state[0] = 0
        elif (cont_state[0] >= -4.8 and cont_state[0] < -2):
            disc_state[0] = 1
        elif (cont_state[0] >= -2 and cont_state[0] < 0):
            disc_state[0] = 2
        elif (cont_state[0] >= 0 and cont_state[0] < 2):
            disc_state[0] = 3
        elif (cont_state[0] >= 2 and cont_state[0] < 4.8):
            disc_state[0] = 4
        else:
            disc_state[0] = 5

        # cont_state[1] is between -inf to +inf
        if (cont_state[1] < -3.4):
            disc_state[1] = 0
        elif (cont_state[1] >= -3.4 and cont_state[1] < -1):
            disc_state[1] = 1
        elif (cont_state[1] >= -1 and cont_state[1] < 0):
            disc_state[1] = 2
        elif (cont_state[1] >= 0 and cont_state[1] < 1):
            disc_state[1] = 3
        elif (cont_state[1] >= 1 and cont_state[1] < 3.4):
            disc_state[1] = 4
        else:
            disc_state[1] = 5

        # cont_state[2] is between -0.41 to 0.41
        if (cont_state[2] < -0.41):
            disc_state[2] = 0
        elif (cont_state[2] >= -0.41 and cont_state[2] < -0.2):
            disc_state[2] = 1
        elif (cont_state[2] >= -0.2 and cont_state[2] < 0):
            disc_state[2] = 2
        elif (cont_state[2] >= 0 and cont_state[2] < 2):
            disc_state[2] = 3
        elif (cont_state[2] >= 2 and cont_state[2] < 0.41):
            disc_state[2] = 4
        else:
            disc_state[2] = 5

        # cont_state[3] is between -inf to +inf
        if (cont_state[3] < -3.4):
            disc_state[3] = 0
        elif (cont_state[3] >= -3.4 and cont_state[3] < -3):
            disc_state[3] = 1
        elif (cont_state[3] >= -3 and cont_state[3] < -2.4):
            disc_state[3] = 2
        elif (cont_state[3] >= -2.4 and cont_state[3] < -1.8):
            disc_state[3] = 3
        elif (cont_state[3] >= -1.8 and cont_state[3] < -1.2):
            disc_state[3] = 4
        elif (cont_state[3] >= -1.2 and cont_state[3] < -0.6):
            disc_state[3] = 5
        elif (cont_state[3] >= -0.6 and cont_state[3] < 0):
            disc_state[3] = 6
        elif (cont_state[3] >= 0 and cont_state[3] < 0.6):
            disc_state[3] = 7
        elif (cont_state[3] >= 0.6 and cont_state[3] < 1.2):
            disc_state[3] = 8
        elif (cont_state[3] >= 1.2 and cont_state[3] < 1.8):
            disc_state[3] = 9
        elif (cont_state[3] >= 1.8 and cont_state[3] < 2.4):
            disc_state[3] = 10
        elif (cont_state[3] >= 2.4 and cont_state[3] < 3):
            disc_state[3] = 11
        else:
            disc_state[3] = 12

        disc_state = [int(disc) for disc in disc_state]

        return tuple(disc_state)

    def agent_train(self, n_episodes):
        episode = 0
        track_max = 0
        track_max_episode = 1
        max_mean_reward = 0
        max_total_reward = 0
        max_reward_count = 0

        # training of agent will be done for n_episodes
        # or till the agent's reward for an episode is 200 for 10 consecutive episodes
        # agent is trained so that with the help of the q value table, best possible actions are chosen
        while max_mean_reward < 200 and episode < n_episodes:
            done = False
            max_reward = 0
            episode += 1

            # converting the initial continuous state to discrete state
            current_state = self.continuous_to_discrete_state(self.env.reset()[0]) # since cont_state is a tuple with state array and dict
            while not done:
                # exploration vs exploitation
                # exploration - if a random number between 0 to 1 is less than epsilon which is decaying at the rate 0.995
                # choosing a random action left or right
                # exploitation - choosing an action from the q-table for corresponding state

                if np.random.random() < self.epsilon:
                    # print("chose random action ")
                    action = np.random.randint(0, self.env.action_space.n)
                else:
                    # print("chose best action ")
                    # print(self.q_table[current_state])
                    action = np.argmax(self.q_table[current_state])

                    # taking action in the environment - produces next state with reward

                next_state, reward, done, trunc, info = self.env.step(action)

                next_state = self.continuous_to_discrete_state(next_state)

                # updating q value of the current state in the q value table
                # q(c_s,a) = q(c_s,a) + alpha(r(c_s,a) + gamma*max(q(n_s)) - q(c_s,a))
                # alpha is the learning rate and gamma is the discount factor for future rewards

                self.q_table[current_state + (action,)] = self.q_table[current_state + (action,)] + \
                                                          self.lr * \
                                                          (reward \
                                                           + (self.discount * np.max(self.q_table[next_state])) \
                                                           - self.q_table[current_state + (action,)])

                current_state = next_state

                max_reward += reward

            # epsilon value is decreased so that the agent will choose best action after exploration of states
            self.decay_epsilon()

            if episode % 1000 == 0 or max_reward > 145:
                print("Reward {} at episode {}".format(max_reward, episode))

            if max_reward >= 200:
                max_reward_count += 1

            if max_reward > track_max:
                track_max = max_reward
                track_max_episode = episode

            max_total_reward += max_reward

            if episode % 10 == 0:
                max_mean_reward = max_total_reward / 10
                max_total_reward = 0

            '''if self.epsilon > 0.1:
                print("Episode: {} ==> Epsilon: {}".format(episode, self.epsilon))'''

        print("\nQ Learning - Training ends at episode {}".format(episode))
        print("Total number of Maximum rewards (200 timesteps) achieved during training: " + str(max_reward_count))
        print("Maximum reward achieved during training: {} at episode {}".format(track_max, track_max_episode))