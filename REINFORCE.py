import numpy as np
import gym
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(Policy, self).__init__()
        # construct neural net
        self.layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_layers[i]))
            else:
                self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*self.layers)
        # reset policy, needs to be done before each new episode
        self.onpolicy_reset()
        # nn.Module, sets mode to training (vs validation, where no exploration)
        self.train() 


    def onpolicy_reset(self):
        """
            resets the log probability (used for calculating the loss) and
            the reward history (used for computing the discounted reward at 
            each step) of the policy.
        """
        self.log_prob = []
        self.rewards = []


    def forward(self, x):
        """
            completes one forward pass of the network with the observation state x.
        """
        return self.model(x)


    def act(self, state):
        """
            computes an action from the current policy, given the current state.

            :return: sampled action
        """
        # map state to tensor
        x = torch.from_numpy(state.astype(np.float32))
        # a single forward pass
        frd_pass = self.forward(x)
        # using Categorical distribution, we compute the prob. dist.
        pd = Categorical(logits=frd_pass)
        # sampling the prob. dist. gives us an action
        action = pd.sample()
        # apply the log prob of the sampled action, used for loss
        self.log_prob.append(pd.log_prob(action))
        return action.item()


def main(env_name, hidden_layers, learning_rate, discount_rate):
    """
        defines the environment, initializes the policy, simulates trajectories 
        of each episode, computes the loss and updates the policy.
    """
    env = gym.make(env_name)
    # if the env space is discrete, we need to use 'n' instead because 'shape returns empty
    try:
        input_dim = env.observation_space.shape[0]
    except:
        input_dim = env.observation_space.n
    try:
        output_dim = env.action_space.shape[0]
    except:
        output_dim = env.action_space.n
    # construct policy
    pi  = Policy(input_dim, hidden_layers, output_dim)
    # construct optimizer
    opt = optim.Adam(pi.parameters(), lr=learning_rate)
    # for each episode...
    for epi in range(EPISODE_NUM):
        state = env.reset()
        # sample a trajectory
        for t in range(MAX_TIMESTEP):
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
        # compute discounted rewards
        T = len(pi.rewards)
        rewards = np.empty(T, dtype=np.float32)
        next_rew = 0.0
        for t in range(T):
            next_rew = pi.rewards[t] + discount_rate * next_rew
            rewards[t] = next_rew
        rewards = torch.tensor(rewards)
        # gradient loss
        log_probs = torch.stack(pi.log_prob)
        loss = torch.sum( -log_probs * rewards )
        # update optimizer
        opt.zero_grad()
        loss.backward()
        opt.step()
        # print out episode stats
        print("\nEpisode: {}\nLoss:    {}\nTotal Reward: {}".format(epi, loss, sum(pi.rewards)))
        # checking if solved
        if sum(pi.rewards) > REW_THRWSHOLD:
            print("SOLVED!")
            break
        # reseting policy for next episode
        pi.onpolicy_reset()
    

if __name__ == '__main__':
    EPISODE_NUM = 1000
    MAX_TIMESTEP = 200
    REW_THRWSHOLD = 200

    main('CartPole-v0', [64], 0.9, 0.99)