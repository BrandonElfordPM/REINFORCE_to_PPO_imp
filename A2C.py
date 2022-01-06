import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

import gym

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

import stable_baselines3 as sb3

from collections import OrderedDict

##########

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, continuous_actions=False, action_std_init=1.0, device='cpu'):
        super(ActorCriticPolicy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_layers is not None:
            pf_arch = hidden_layers["policy"]
            vf_arch = hidden_layers["value"]
        layers = self.compile_layers(input_dim, output_dim, pf_arch)
        if continuous_actions:
            layers['af'] = nn.Tanh()
        else:
            layers['af'] = nn.Softmax(dim=-1)
        self.actor  = nn.Sequential( layers )
        self.critic = nn.Sequential( self.compile_layers(input_dim, 1, vf_arch) )
        self.device = device

    def compile_layers(self, input_dim, output_dim, arch):
        """
            given a list of layers, this builds the full policy (with activation 
            functions between layers)
        """
        net_len = len(arch)
        layers = OrderedDict()
        prev_size = input_dim[0]
        for idx in range(net_len):
            layers['lin'+str(idx)]  = nn.Linear(prev_size, arch[idx])
            layers['relu'+str(idx)] = nn.ReLU()
            prev_size = arch[idx]
        layers['out'] = nn.Linear(arch[-1], output_dim)
        return layers

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
        X = torch.FloatTensor(x)
        return self.critic(X), self.actor(X)

    def act(self, obs):
        """
            computes the action distribution and return an action and log probability.
        """
        value, act_probs = self.forward(obs)
        dist = Categorical(act_probs)
        action = dist.sample()
        return action.item()

    def eval(self, obs, action):
        """
            evaluates an action given a current state, returns log probability, 
            critic's action score and entropy in action distribution
        """
        value, act_probs = self.forward(obs)
        dist = Categorical(act_probs)
        log_probs = dist.log_prob(action).view(-1,1)
        entropy = dist.entropy().mean()
        return value, log_probs, entropy

##########

class RolloutBuffer:
    def __init__(self):
        self.actions      = []
        self.observations = []
        self.rewards      = []
        self.cuml_rewards = []
        self.episode_rews = []
        self.dones        = []

    def reset(self):
        self.actions      = []
        self.observations = []
        self.rewards      = []
        self.cuml_rewards = []
        self.episode_rews = []
        self.dones        = []

##########

class A2C:
    def __init__(self, env, policy=None, policy_kwargs=None, optimizer=None, hyper_params=None):
        # Environment
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        # Hyper parameters
        if hyper_params is not None:
            self._lr           = hyper_params["lr"]           if "lr"           in hyper_params else 3e-3
            self._gamma        = hyper_params["gamma"]        if "gamma"        in hyper_params else 0.99
            self._lambda       = hyper_params["lambda"]       if "lambda"       in hyper_params else 0.95
            self._epsilon      = hyper_params["epsilon"]      if "epsilon"      in hyper_params else 0.2  #check default
            self._entropy_coef = hyper_params["entropy_coef"] if "entropy_coef" in hyper_params else 0.01
            self._critic_coef  = hyper_params["critic_coef"]  if "critic_coef"  in hyper_params else 0.5
            self._clip         = hyper_params["clip"]         if "clip"         in hyper_params else 0.5
            self._batch_size   = hyper_params["batch_size"]   if "batch_size"   in hyper_params else 64
        else:
            self._lr           = 3e-3
            self._gamma        = 0.99
            self._lambda       = 0.95
            self._epsilon      = 0.2  # check default value
            self._entropy_coef = 0.01
            self._critic_coef  = 0.1
            self._clip         = 0.9
            self._batch_size   = 64
        # Policy
        if type(self.observation_space) is gym.spaces.Discrete:
            input_dim = self.observation_space.n
        else:
            input_dim  = self.observation_space.shape
        if type(self.action_space) is gym.spaces.Discrete:
            output_dim = self.action_space.n
        else:
            output_dim = self.action_space.shape
        hidden_layers = policy_kwargs["network_arch"] if ( (policy_kwargs is not None) and ("network_arch" in policy_kwargs) ) else None
        
        if policy is None:
            self.policy = ActorCriticPolicy(input_dim, 
                                            output_dim, 
                                            hidden_layers)
        else:
            self.policy = policy
        self.old_policy = ActorCriticPolicy(input_dim, 
                                            output_dim, 
                                            hidden_layers)
        self.old_policy.load_state_dict(self.policy.state_dict())
        # Optimizer
        self.optimizer = optimizer if optimizer is not None else Adam(self.policy.actor.parameters(), lr=self._lr, eps=self._epsilon)
        # RolloutBuffer
        self.rollout = RolloutBuffer()

    def learn(self):
        acts = torch.FloatTensor(self.rollout.actions)
        obs  = torch.FloatTensor(self.rollout.observations)
        rews = torch.FloatTensor(self.rollout.cuml_rewards)
        # compute advantages
        values, log_probs, entropy = self.policy.eval(obs, acts)
        advantages = rews - values.reshape(-1) # THIS IS WRONG?
        # compute losses                       # Loss is constant, reward improves ?
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(log_probs*advantages).mean()
        loss = (self._critic_coef * critic_loss) + actor_loss*(self._entropy_coef*entropy)
        # update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.policy.actor.parameters(), self._clip)
        self.optimizer.step()
        return values.mean().item(), critic_loss.item(), actor_loss.item(), loss.item()

    def compute_cumlative_rewards(self):
        cuml_rewards = []
        episode_rewards = []
        rewards = torch.FloatTensor(self.rollout.rewards)
        observations = torch.FloatTensor(self.rollout.observations)
        dones = torch.FloatTensor(self.rollout.dones)
        if dones[-1] == True:
            curr_rew = rewards[-1]
        else:
            curr_rew = self.policy.critic(observations[-1].unsqueeze(0)) 
        cuml_rewards.append(curr_rew)
        for i in reversed(range(0,len(rewards)-1)):
            if dones[i]==True:
                curr_rew = rewards[i]
                episode_rewards.append(curr_rew)
            else:
                curr_rew = rewards[i] + self._gamma*curr_rew
            cuml_rewards.append(curr_rew)
        cuml_rewards.reverse()
        self.rollout.cuml_rewards = cuml_rewards

    def get_action(self, obs):
        actions = self.policy.act(torch.FloatTensor(obs.reshape(1,-1)))
        return actions

    def save(self, model_path):
        torch.save([self.policy.state_dict(),self.optimizer.state_dict()], model_path)

    def load(self, model_path):
        loaded = torch.load(model_path)
        self.policy.load_state_dict(loaded[0])
        self.optimizer.load_state_dict(loaded[1])


##########

def plot(data, frame_idx):
    plt.figure(figsize=(20,5))
    if data['episode_rewards']:
        ax = plt.subplot(121)
        ax = plt.gca()
        avg_score = np.mean(data['episode_rewards'][-100:])
        plt.title("Frame: {}   Average Score: {:.2f}".format(frame_idx, avg_score))
        plt.grid()
        plt.plot(data['episode_rewards'])
    if data['values']:
        ax = plt.subplot(122)
        avg_value = np.mean(data['values'][-1000:])
        plt.title("Frame: {}   Average Value: {:.2f}".format(frame_idx, avg_value))
        plt.plot(data['values'])
    plt.show()

def main():
    env = gym.make("CartPole-v1")

    timesteps = 1e6
    curr_step = 0
    episode_rew = 0
    episode_rews = [0]
    values = []
    loss = []
    next_log_step = 0
    log_step = 50000

    lr = 3e-3
    batch_size = 16
    
    obs = env.reset()

    hyperparameters = {
                        'lr':           lr,
                        "gamma":        0.99,
                        "lambda":       0.95,
                        "epsilon":      0.2,
                        "entropy_coef": 0.01,
                        "critic_coef":  0.5,
                        "clip":         0.5,
                        "batch_size":   batch_size 
                      }

    policy_kwargs = { "network_arch": { "policy": [16], 
                                        "value":  [16] } 
                                        }
    model = A2C(env, policy_kwargs=policy_kwargs, hyper_params=hyperparameters)

    print("Training...")
    while curr_step < timesteps:
        for i in range(batch_size):
            action = model.policy.act(obs)
            next_obs, rew, done, _ = env.step(action)
            episode_rew += rew

            model.rollout.actions.append(action)
            model.rollout.observations.append(obs)
            model.rollout.rewards.append(rew)
            model.rollout.dones.append(done)

            if done:
                obs = env.reset()
                episode_rews.append(episode_rew)
                episode_rew = 0
            else:
                obs = next_obs

        curr_step += batch_size

        model.compute_cumlative_rewards()
        #episode_rews.extend( epi_rews )

        value, val_loss, pol_loss, tot_loss = model.learn()
        loss.append(tot_loss)
        if curr_step % 100 == 0:
            values.append(value)

        if curr_step >= next_log_step:
            # LOG Actor_loss, episode_rew, entropy, value_loss
            print("Timestep: {:8d}    Avg Epi Rew: {:6.3f}    Val Loss: {:6.3f}    Pol Loss: {:5.3f}    Avg Val: {:.3f}".format(curr_step, np.mean(episode_rews), val_loss, pol_loss, np.mean(values)))
            next_log_step += log_step

        model.rollout.reset()
    
    plot({'episode_rewards': episode_rews, 'values': values}, curr_step)

    model.save('a2c_model')

    del model
    
    print("Validating...")
    model = A2C(env, policy_kwargs=policy_kwargs)
    model.load('a2c_model')

    obs = env.reset()
    done = False
    while not done:
        act = model.get_action(obs)
        obs, rew, done, _ = env.step(act)
        env.render()


if __name__ == '__main__':
    main()