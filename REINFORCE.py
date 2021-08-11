import numpy as np
import gym
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


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
        return self.model(x)


    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        frd_pass = self.forward(x)
        pd = Categorical(logits=frd_pass)

        action = pd.sample()
        
        self.log_prob.append(pd.log_prob(action))

        return action.item()


def main(env_name, hidden_layers, learning_rate, discount_rate):
    """

    """
    env = gym.make(env_name)

    try:
        input_dim = env.observation_space.shape[0]
    except:
        input_dim = env.observation_space.n

    try:
        output_dim = env.action_space.shape[0]
    except:
        output_dim = env.action_space.n

    pi  = Policy(input_dim, hidden_layers, output_dim)
    opt = optim.Adam(pi.parameters(), lr=learning_rate)

    for epi in range(EPISODE_NUM):
        state = env.reset()
        # sample a trajectory
        for t in range(MAX_TIMESTEP):
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            print("State: {}".format(state))
            if done:
                print("Env ended\nState: {}".format(state))
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