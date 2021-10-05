import numpy as np
import gym
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticPolicy(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    def act(self):
        pass

    def eval(self):
        pass


class RolloutBuffer:
    def __init__(self):
        pass

    def reset(self):
        pass


class PPO:
    def __init__(self):
        pass

    def get_action(self):
        pass

    def update(self):
        pass

    def save(self):
        pass

    def load(self):
        pass