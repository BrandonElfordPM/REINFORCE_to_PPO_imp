import numpy as np
import gym
import torch 
import torch.nn as nn
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(Policy, self).__init__()
        self.layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_layers[i]))
            else:
                self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.model = nn.Sequential(*self.layers)
        self.onpolicy_reset()
        self.train() # nn.Module, sets mode to training (vs validation, where no exploration)


    def onpolicy_reset(self):
        self.log_prob = []
        self.rewards = []


    def forward(self, x):
        pass


    def act(self, state):
        pass


def train(policy, optimizer):
    pass


if __name__ == '__main__':
    learning_rate = 0.9
    env = gym.make('CartPole-v0')

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    hidden_layers = [32]

    pi = Policy(input_dim, hidden_layers, output_dim)

    EPISODE_NUM = 100
    for epi in range(EPISODE_NUM):
        trajectory = []
        # get episode trajectory (of thruples)
        # reset on_policy
        for i in range(len(trajectory)):
            rew = 0
            # calculate discounted reward, append
            # compute gradient
        # update weights
    