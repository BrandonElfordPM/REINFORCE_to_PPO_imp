import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

import gym

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical

###

class Policy(nn.Module):
    def __init__(self, 
                 input_dim: int=1, 
                 hidden_layers: list= [8], 
                 output_dim: int=1):
        super().__init__()
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

class REINFORCE:
    def __init__(self, 
                 env: gym.Env, 
                 policy: nn.Module = None, 
                 policy_kwargs: dict = {}, 
                 optimizer: torch.optim = None, 
                 hyper_params: dict = {},
                 model_name: str = None,
                 ):
        self.env = env
        # if the env space is discrete, we need to use 'n' instead because 'shape returns empty
        try:
            self.observation_space = env.observation_space.shape[0]
        except:
            self.observation_space = env.observation_space.n
        try:
            self.action_space = env.action_space.shape[0]
        except:
            self.action_space = env.action_space.n
        # hyperparams
        self._lr            = hyper_params['lr']            if 'lr'            in hyper_params else 3e-3
        self._gamma         = hyper_params['gamma']         if 'gamma'         in hyper_params else 0.99
        self._hidden_layers = hyper_params['hidden_layers'] if 'hidden_layers' in hyper_params else [32]
        # construct policy
        if not policy:
            self.policy = Policy(self.observation_space, 
                                 self._hidden_layers, 
                                 self.action_space)
        else:
            self.policy = policy
        # construct optimizer
        if not optimizer:
            self.optimizer = optim.Adam(self.policy.parameters(), 
                                        lr=self._lr)
        else:
            self.optimizer = optimizer
        # load weights
        if model_name:
            self.load(file_name=model_name)

        self.episode_rews = []

    def do_epoch(self, 
                 max_timestep: int = 5e2):
        state = self.env.reset()
        # sample a trajectory
        for t in range(max_timestep):
            action = self.policy.act(state)
            state, reward, done, _ = self.env.step(action) # [action] for Pendulum, action otherwise
            self.policy.rewards.append(reward)
            # env.render()
            if done:
                break
        # convert step rewards to cumulative rewards
        rewards = self.compute_cumulative_rewards()
        # gradient loss
        log_probs = torch.stack(self.policy.log_prob)
        loss = torch.sum( -log_probs * rewards )
        # update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # recording episode rewards
        self.episode_rews.append(sum(self.policy.rewards))
        # print out episode stats
        avg_epi_rew = np.mean(self.episode_rews[-10:])
        return avg_epi_rew, loss

    def do_n_epochs(self, 
                    num_episodes: int = 1e3,
                    max_timestep: int = 5e2,
                    rew_threshold: int = 5e2 // (9/10),
                    save_model: bool = False,
                    model_name: str = 'model_ckpt'
                    ):
        # for each episode...
        for epi in range(num_episodes):
            avg_epi_rew, loss = self.do_epoch(max_timestep)
            if epi % 100 == 0:
                print("\nEpisode: {}\nLoss:    {}\nAvg Epi Rew: {}".format(epi, loss, avg_epi_rew))
            # checking if solved
            if avg_epi_rew > rew_threshold:
                print("SOLVED")
                break
            # reseting policy for next episode
            self.policy.reset()
        if save_model:
            print('\nSaving...\n')
            self.save(model_name)

    def compute_cumulative_rewards(self):
        # compute cumulative rewards
        T = len(self.policy.rewards)
        rewards = np.empty(T, dtype=np.float32)
        next_rew = 0.0
        for t in range(T):
            next_rew = self.policy.rewards[t] + self._gamma * next_rew
            rewards[t] = next_rew
        rewards = torch.tensor(rewards)
        return rewards

    def save(self,
             env_name: str,
             model_name: str='model_ckpt',
             ):
        file_name = self.env.unwrapped.spec.id+'-'+model_name
        torch.save({'model_state_dict': self.policy.model.state_dict(), 
                    'optim_state_dict': self.optimizer.state_dict()},
                   model_name)

    def load(self,
             file_name: str = 'module_ckpt',
             ):
        state_dict = torch.load(file_name)
        self.policy.model.load_state_dict(state_dict['model_state_dict'])
        self.optim.load_state_dict(state_dict['optim_state_dict'])

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

###

def train(env_name):
    num_episodes = 10000
    max_timestep = 1000
    rew_threshold = 0.9 * max_timestep
    # create environment
    env = gym.make(env_name)
    env._max_episode_steps = max_timestep
    # algo
    hyper_params = {'hidden_layers': [16,16]}
    algo = REINFORCE(env, hyper_params=hyper_params)
    algo.do_n_epochs(num_episodes=num_episodes, 
                     max_timestep=max_timestep, 
                     rew_threshold=rew_threshold
                     )
    return algo

def valid(algo):
    policy = algo.policy
    policy.eval()
    env = algo.env
    frames = []
    policy.eval()
    state = env.reset()
    while True:
        action = policy.act(state)
        state, rew, done, _ = env.step(action)
        env.render()
        frames.append(env.render(mode="rgb_array"))
        if done:
            break
    file_name = env.unwrapped.spec.id+'_REINFORCE.gif'
    save_frames_as_gif(frames, path='images/', filename=file_name)

###

if __name__ == '__main__':
    env_name = 'CartPole-v1' # solved
    # env_name = 'Acrobot-v1'  # solved
    
    algo = train(env_name)

    valid(algo)