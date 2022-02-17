import numpy as np
import gym
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

import stable_baselines3 as sb3

from collections import OrderedDict


class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, continuous_actions=False, action_std_init=1.0, device='cpu'):
        super(ActorCriticPolicy, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.continuous_actions = continuous_actions
        if self.continuous_actions:
            self.action_var = torch.full((output_dim,), 
                                         action_std_init**2).to(device)

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
            action_mean = self.forward(obs)
            dist = MultivariateNormal(action_mean, torch.diag(self.action_var).unsqueeze(dim=0))
        else:
            action_prob = self.actor(obs)
            dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def eval(self, obs, action):
        if self.continuous_actions:
            action_mean = self.forward(obs)
            action_cov_matrix = torch.diag_embed(self.action_var)
            dist = MultivariateNormal(action_mean, covariance_matrix=action_cov_matrix)
        else:
            act_probs = self.forward(torch.FloatTensor(obs))
            logits = torch.log(act_probs)
            dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(torch.FloatTensor(action))
        return action_logprobs, self.critic(torch.FloatTensor(obs)), dist.entropy()

    def compile_layers(self, input_dim, output_dim, arch):
        net_len = len(arch)
        layers = OrderedDict()
        prev_size = input_dim[0]
        for idx in range(net_len):
            layers['lin'+str(idx)]  = nn.Linear(prev_size, arch[idx])
            layers['relu'+str(idx)] = nn.ReLU()
            prev_size = arch[idx]
        layers['out'] = nn.Linear(arch[-1], output_dim)
        return layers


class RolloutBuffer:
    def __init__(self):
        self.actions      = []
        self.observations = []
        self.rewards      = []
        self.episode_rews = []
        self.dones        = []
        self.log_probs    = []
        self.advantages   = []
        self.ratios       = []

    def reset(self):
        self.actions      = []
        self.observations = []
        self.rewards      = []
        self.episode_rews = []
        self.dones        = []
        self.log_probs    = []
        self.advantages   = []
        self.ratios       = []


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
        self.optimizer = optimizer if optimizer is not None else Adam(self.policy.parameters(), lr=self._lr, eps=self._epsilon)
        # RolloutBuffer
        self.rollout = RolloutBuffer()

    def get_action(self, obs):
        actions, log_prob = self.old_policy.act(torch.FloatTensor(obs.reshape(1,-1))) # actions == 0, always
        self.rollout.log_probs.append(log_prob)
        return actions.cpu().numpy().flatten()

    def update(self):
        obs = self.env.reset()
        # compute trajectories in batches
        for i in range(self._batch_size):
            action = self.get_action(obs)
            self.rollout.actions.append(action)
            obs, rew, done, _ = self.env.step(action[0])
            self.rollout.observations.append(obs)
            self.rollout.rewards.append(rew)
            self.rollout.dones.append(done)
            if done:
                obs = self.env.reset()

        # combine rewards
        rewards = []
        discounted_rew = 0
        for rew, done in zip(self.rollout.rewards, self.rollout.dones):
            if done:
                self.rollout.episode_rews.append(discounted_rew)
                discounted_rew = 0
            discounted_rew = rew + self._gamma*discounted_rew
            rewards.insert(0, discounted_rew)
        # normalize rewards
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (np.max([rewards.std().item(), 1e-10]))
        # compute log probabilities 
        log_probs, state_vals, dist_entropy = self.policy.eval(self.rollout.observations, self.rollout.actions)
        # compute advantages
        advantages = torch.FloatTensor(rewards) - state_vals
        self.rollout.advantages.append(advantages)
        # compute ratios
        state_vals = torch.squeeze(state_vals)
        ratio = torch.exp(log_probs - torch.FloatTensor(self.rollout.log_probs).detach())
        self.rollout.ratios.append(ratio)
        # compute first surrogate loss
        loss_1 = ratio * advantages
        # compute second surrogate loss
        loss_2 = torch.clip(ratio, 1-self._epsilon, 1+self._epsilon) * advantages
        # clipping loss
        actor_loss = -torch.min( loss_1, loss_2 )
        MSE_loss = nn.MSELoss()
        critic_loss = ((MSE_loss(torch.FloatTensor(rewards), state_vals))/2.0) - 0.01 * dist_entropy
        loss = actor_loss + critic_loss
        # use optimizer
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # print("Actor loss:  ", actor_loss)
        # print("Critic loss: ", critic_loss)

    def save(self, model_path='/./model'):
        torch.save( self.policy_old.state_dict(), model_path )

    def load(self, model_path):
        self.policy.load_state_dict(torch.load(model_path))
        self.old_policy.load_state_dict(torch.load(model_path))


def train_test_sb3():
    env = gym.make("CartPole-v1")
    policy_kwargs = { "net_arch": [{ "pi": [32, 32], 
                                     "vf": [32, 32] } ]
                                    }
    ppo_sb3 = sb3.PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
    print("\nTRAINING...")
    print("SB3...")
    ppo_sb3.learn(total_timesteps=10000)
    print("\nTESTING...")
    print("SB3...")
    time_step = 0
    total_reward = 0 
    obs = env.reset()
    while True:
        time_step += 1
        act, _ = ppo_sb3.predict(obs)
        obs, rew, done, _ = env.step(act)
        total_reward += rew
        env.render()
        if done:
            print("Test done, total reward {}".format(total_reward))
            break

def train_mine(env, model, model_path=None):
    print("\nTRAINING...")
    print("MINE...")
    avg_reward = 0
    for i in range(num_episodes):
        if i % 10 == 0:
            print("Episode ", i, "\tAvg Rew ", avg_reward)
        for _ in range(num_epochs):
            model.update()
            # update old policy
            model.old_policy.load_state_dict(model.policy.state_dict())
            avg_reward = np.mean(model.rollout.episode_rews)
            # for i in range(64):
            #     print('Actions\n',ppo.rollout.actions[i])
            #     print('Obs\n',ppo.rollout.observations[i])
            #     print('Rewards\n',ppo.rollout.rewards[i])
            #     print(ppo.rollout.dones[i])
            #     print(ppo.rollout.log_probs[i])
            #     print(ppo.rollout.advantages[0][i])
            #     print(ppo.rollout.ratios[0][i])
            #     print('---')
            # print("++++++++++++++++++")
            model.rollout.reset()
    model.save(model_path=model_path)

def test_mine(env, model, model_path):
    print("\nTESTING...")
    print("MINE...")
    model.load(model_path)
    time_step = 0
    total_reward = 0 
    obs = env.reset()
    while True:
        time_step += 1
        act = model.get_action(obs)
        obs, rew, done, _ = env.step(act[0])
        total_reward += rew
        env.render()
        if done:
            print("Test done, total reward {}".format(total_reward))
            break

if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    policy_kwargs = { "network_arch": { "policy": [32, 32], 
                                        "value":  [32, 32] } 
                                        }
    model = PPO(env, policy_kwargs=policy_kwargs)
    num_epochs = 100
    num_episodes = 100
    model_path = 'model'
    
    train_mine(env, model, model_path)
    test_mine(env, model_path)
    # train_test_sb3()


### DOESNT LEARN ###
# need to debug the learning, doesn't seem to be ANY learning at all