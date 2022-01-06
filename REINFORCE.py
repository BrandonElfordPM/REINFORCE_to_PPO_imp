import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

import gym

import torch 
import torch.nn as nn
import torch.nn.functional as F
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
        print(self.model)
        # reset policy, needs to be done before each new episode
        self.reset()
        # nn.Module, sets mode to training (vs validation, where no exploration)
        self.train() 

    def reset(self):
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
        # forward propogation
        x = self.model(x)
        # applying softmax activation to output because the env is discrete
        return F.softmax(x, dim=1)

    def act(self, state):
        """
            returns an action from the probability distribution, 
            also appends the log probability to the model.

            :return: action
        """
        # map state to tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        # forward pass returns action probabilities
        probs = self(state)
        # initialize the action distribution
        dist = Categorical(probs)
        # sample the distribution to return an action
        action = dist.sample()
        # uodate the log_prob list
        self.log_prob.append(dist.log_prob(action))
        return action.item()

###

def REINFORCE(env, hidden_layers, learning_rate, discount_rate):
    """
        defines the environment, initializes the policy, simulates trajectories 
        of each episode, computes the loss and updates the policy.
    """
    epi_rewards = []
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
            # env.render()
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
        # recording episode rewards
        epi_rewards.append(sum(pi.rewards))
        # print out episode stats
        avg_epi_rew = np.mean(epi_rewards[-10:])
        if epi % 100 == 0:
            print("\nEpisode: {}\nLoss:    {}\nAvg Epi Rew: {}".format(epi, loss, avg_epi_rew))
        # checking if solved
        if avg_epi_rew > REW_THRWSHOLD:
            print("SOLVED")
            break
        # reseting policy for next episode
        pi.reset()
    return pi
    
###

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow', fps=60)

def valid(pi, env_name):
    env = gym.make(env_name)
    frames = []
    pi.eval()
    state = env.reset()
    while True:
        action = pi.act(state)
        state, rew, done, _ = env.step(action)
        env.render()
        frames.append(env.render(mode="rgb_array"))
        if done:
            break
    file_name = env_name+'_REINFORCE.gif'
    save_frames_as_gif(frames, filename=file_name)

###

if __name__ == '__main__':
    EPISODE_NUM = 10000
    MAX_TIMESTEP = 1000
    REW_THRWSHOLD = 0.9 * MAX_TIMESTEP

    env_name = 'CartPole-v1' # solved
    # env_name = 'Acrobot-v1'  # solved
    # env_name = 'MountainCar-v0'
    hidden_layers = [32]
    learning_rate = 0.003
    discount_rate = 0.99

    env = gym.make(env_name)
    env._max_episode_steps = MAX_TIMESTEP

    pi = REINFORCE(env, hidden_layers, learning_rate, discount_rate)

    valid(pi, env_name)