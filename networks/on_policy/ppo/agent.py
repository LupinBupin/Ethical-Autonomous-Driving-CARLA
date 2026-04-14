import os
import numpy as np

import torch
import torch.nn as nn
from encoder_init import EncodeState
from networks.on_policy.ppo.ppo import ActorCritic
from parameters import  *

device = torch.device("cpu")

class Buffer:
    def __init__(self):
         # Batch data
        self.observation = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []

    def clear(self):
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]

class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        
        #self.env = env
        self.obs_dim = 100
        self.action_dim = 2
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 1
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        self.town = town

        self.checkpoint_file_no = 0
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr}])

        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def _prepare_observation(self, obs):
        if obs is None:
            raise ValueError("Received None observation in PPOAgent.get_action()")

        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.detach().to(dtype=torch.float32, device=device)
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

        return obs_tensor.flatten()


    def _is_valid_state_dict(self, state_dict):
        for tensor in state_dict.values():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return False
        return True


    def _has_non_finite(self, tensor):
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()


    def get_action(self, obs, train):
        obs_tensor = self._prepare_observation(obs)
        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

        with torch.no_grad():
            action, logprob = self.old_policy.get_action_and_log_prob(obs_tensor)
        if train:
            self.memory.observation.append(obs_tensor)
            self.memory.actions.append(action.detach().to(device))
            self.memory.log_probs.append(logprob.detach().to(device))

        return action.detach().cpu().numpy().flatten()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std


    # def learn(self):

    #     print("LEARN START")
    #     print("rewards:", len(self.memory.rewards))
    #     print("dones:", len(self.memory.dones))
    #     print("observations:", len(self.memory.observation))
    #     print("actions:", len(self.memory.actions))
    #     print("log_probs:", len(self.memory.log_probs))
    #     print("device:", device)

    #     # Monte Carlo estimate of returns
    #     rewards = []
    #     discounted_reward = 0
    #     for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
    #         if is_terminal:
    #             discounted_reward = 0
    #         discounted_reward = reward + (self.gamma * discounted_reward)
    #         rewards.insert(0, discounted_reward)

    #     if len(rewards) == 0:
    #         print("LEARN ERROR: no rewards collected")
    #         return    
    #     # Normalizing the rewards
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    #     if torch.isnan(rewards).any():
    #         print("LEARN ERROR: NaN in rewards")
    #         return
    #     # convert list to tensor
    #     old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
    #     old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
    #     old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)

        
    #     # Optimize policy for K epochs
    #     for _ in range(self.n_updates_per_iteration):

    #         # Evaluating old actions and values
    #         logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)

    #         # match values tensor dimensions with rewards tensor
    #         values = torch.squeeze(values)
            
    #         # Finding the ratio (pi_theta / pi_theta__old)
    #         ratios = torch.exp(logprobs - old_logprobs.detach())

    #         # Finding Surrogate Loss
    #         advantages = rewards - values.detach()   
    #         surr1 = ratios * advantages
    #         surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

    #         # final loss of clipped objective PPO
    #         loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(values, rewards) - 0.01*dist_entropy
            
    #         # take gradient step
    #         self.optimizer.zero_grad()
    #         loss.mean().backward()
    #         self.optimizer.step()

    #     print("old_states shape:", old_states.shape)
    #     print("old_actions shape:", old_actions.shape)
    #     print("old_logprobs shape:", old_logprobs.shape)
    #     print("rewards shape:", rewards.shape)

    #     self.old_policy.load_state_dict(self.policy.state_dict())
    #     self.memory.clear()

    def learn(self):
        if len(self.memory.rewards) == 0:
            print("Skipping PPO update: empty reward buffer")
            return

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1e6, neginf=-1e6)
        reward_std = rewards.std(unbiased=False)
        rewards = (rewards - rewards.mean()) / (reward_std + 1e-7)

        if self._has_non_finite(rewards):
            print("Skipping PPO update: normalized rewards contain non-finite values")
            self.memory.clear()
            return

        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)
        old_states = torch.nan_to_num(old_states, nan=0.0, posinf=1e6, neginf=-1e6)
        old_actions = torch.nan_to_num(old_actions, nan=0.0, posinf=1.0, neginf=-1.0)
        old_logprobs = torch.nan_to_num(old_logprobs, nan=0.0, posinf=1e6, neginf=-1e6)

        if self._has_non_finite(old_states) or self._has_non_finite(old_actions) or self._has_non_finite(old_logprobs):
            print("Skipping PPO update: rollout buffer contains non-finite values")
            self.memory.clear()
            return

        for _ in range(self.n_updates_per_iteration):
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            if self._has_non_finite(logprobs) or self._has_non_finite(values) or self._has_non_finite(dist_entropy):
                print("Skipping PPO update: policy evaluate() returned non-finite values")
                self.memory.clear()
                return

            values = torch.squeeze(values)

            log_ratios = torch.clamp(logprobs - old_logprobs.detach(), min=-20, max=20)
            ratios = torch.exp(log_ratios)

            advantages = rewards - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(values, rewards) - 0.01*dist_entropy
            if self._has_non_finite(loss):
                print("Skipping PPO update: loss contains non-finite values")
                self.memory.clear()
                return

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            invalid_grad = False
            for param in self.policy.parameters():
                if param.grad is not None and self._has_non_finite(param.grad):
                    invalid_grad = True
                    break

            if invalid_grad:
                print("Skipping PPO update: gradients contain non-finite values")
                self.optimizer.zero_grad()
                self.memory.clear()
                return

            self.optimizer.step()

            if not self._is_valid_state_dict(self.policy.state_dict()):
                print("Skipping PPO checkpoint sync: policy weights became non-finite after optimizer step")
                self.memory.clear()
                return

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    
    def save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        if not self._is_valid_state_dict(self.old_policy.state_dict()):
            print(f"Skipping invalid PPO save: {checkpoint_file}")
            return
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def chkpt_save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        if self.checkpoint_file_no !=0:
            self.checkpoint_file_no -=1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        if not self._is_valid_state_dict(self.old_policy.state_dict()):
            print(f"Skipping invalid PPO checkpoint save: {checkpoint_file}")
            return
        torch.save(self.old_policy.state_dict(), checkpoint_file)
   
    def load(self):
        checkpoint_dir = PPO_CHECKPOINT_DIR + self.town
        checkpoint_count = len(next(os.walk(checkpoint_dir))[2])

        for checkpoint_no in range(checkpoint_count - 1, -1, -1):
            checkpoint_file = checkpoint_dir + "/ppo_policy_" + str(checkpoint_no) + "_.pth"
            state_dict = torch.load(checkpoint_file, map_location=device)

            if not self._is_valid_state_dict(state_dict):
                print(f"Skipping invalid PPO checkpoint: {checkpoint_file}")
                continue

            self.checkpoint_file_no = checkpoint_no
            self.old_policy.load_state_dict(state_dict)
            self.policy.load_state_dict(state_dict)
            print(f"Loaded PPO checkpoint: {checkpoint_file}")
            return

        raise ValueError(f"No valid PPO checkpoints found in {checkpoint_dir}")
            
