import numpy as np
import gym
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

from collections import OrderedDict


class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, continuous_actions=True, action_std_init=None, device='cpu'):
        super(ActorCriticPolicy, self).__init__()

        self.continuous_actions = continuous_actions
        if self.continuous_actions:
            self.action_var = torch.full((output_dim,), action_std_init**2).to(device)

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
        return self.actor(x)

    def act(self, obs):
        if self.continuous_actions:
            action_mean = self.actor(obs)
            dist = MultivariateNormal(action_mean, torch.diag(self.action_var).unsqueeze(dim=0))
        else:
            action_prob = self.actor(obs)
            dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def eval(self, state, action):
        pass

    def compile_layers(self, input_dim, output_dim, arch):
        net_len = len(arch)
        layers = OrderedDict()
        for idx in range(net_len):
            layers['lin'+str(idx)]  = nn.Linear(input_dim, arch[idx])
            layers['relu'+str(idx)] = nn.ReLU()
        layers['out'] = nn.Linear(arch[-1], output_dim)
        return layers


class RolloutBuffer:
    def __init__(self):
        self.actions      = []
        self.observations = []
        self.rewards      = []
        self.dones        = []
        self.log_probs    = []

    def reset(self):
        self.actions      = []
        self.observations = []
        self.rewards      = []
        self.dones        = []
        self.log_probs    = []


class PPO:
    def __init__(self, env, policy=None, policy_kwargs=None, optimizer=None, hyper_params=None):
        # Environment
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        # Hyper parameters
        if hyper_params is not None:
            self._lr         = hyper_params["lr"]         if "lr"         in hyper_params else 3e-3
            self._gamma      = hyper_params["gamma"]      if "gamma"      in hyper_params else 0.99
            self._lambda     = hyper_params["lambda"]     if "lambda"     in hyper_params else 0.95
            self._epsilon    = hyper_params["epsilon"]    if "epsilon"    in hyper_params else 0.2  #check default
            self._batch_size = hyper_params["batch_size"] if "batch_size" in hyper_params else 64
        else:
            self._lr         = 3e-3
            self._gamma      = 0.99
            self._lambda     = 0.95
            self._epsilon    = 0.2 # check default value
            self._batch_size = 64
        # Policy
        input_dim  = self.observation_space.shape
        hidden_layers = policy_kwargs["network_arch"] if ( (policy_kwargs is not None) and ("network_arch" in policy_kwargs) ) else None
        output_dim = self.action_space.shape
        if policy is not None:
            self.policy = ActorCriticPolicy(input_dim, 
                                            output_dim, 
                                            hidden_layers)
        else:
            self.policy = policy
        # Optimizer
        self.optimizer = optimizer if optimizer is not None else Adam(self.policy.parameters(), lr=self._lr, eps=self._epsilon)
        # RolloutBuffer
        self.rollout = RolloutBuffer()

    def get_action(self):
        pass

    def update(self):
        for _ in range(self._batch_size):
            action = self.get_action()
            self.rollout.actions.append(action)
            obs, rew, done, _ = self.env.step(action)
            self.rollout.observations.append(obs)
            self.rollout.rewards.append(rew)
            self.rollout.dones.append(done)
        # compute advantage estimates
        # use optimizer

    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':
    pass