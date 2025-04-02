# src/deeprl/memory/replay_buffer.py
import numpy as np
import torch


class HumanDemonstrationBuffer:
    """
    Buffer for storing human demonstration data.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self):
        """
        Clear the buffer.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def get_data(self, device):
        """
        Get the data as tensors on the specified device.

        Args:
            device (torch.device): Device to put tensors on

        Returns:
            dict: Dictionary of tensors
        """
        return {
            "states": torch.FloatTensor(np.array(self.states)).to(device),
            "actions": torch.LongTensor(np.array(self.actions)).to(device),
            "rewards": torch.FloatTensor(np.array(self.rewards)).to(device),
            "next_states": torch.FloatTensor(np.array(self.next_states)).to(device),
            "dones": torch.FloatTensor(np.array(self.dones)).to(device),
        }

    def compute_returns(self, gamma=0.99):
        """
        Compute returns for all stored trajectories.

        Args:
            gamma (float): Discount factor

        Returns:
            numpy.ndarray: Array of returns
        """
        returns = []
        next_return = 0

        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                next_return = 0
            next_return = r + gamma * next_return
            returns.insert(0, next_return)

        return np.array(returns)

    def __len__(self):
        return len(self.states)


class RolloutBuffer:
    """
    Buffer for storing rollout data from the agent.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        """
        Add a transition to the buffer.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """
        Clear the buffer.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Args:
            last_value (float): Value of the last state
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter

        Returns:
            tuple: (returns, advantages)
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values)

        return returns, advantages

    def get_data(self, device):
        """
        Get the data as tensors on the specified device.

        Args:
            device (torch.device): Device to put tensors on

        Returns:
            dict: Dictionary of tensors
        """
        return {
            "states": torch.FloatTensor(np.array(self.states)).to(device),
            "actions": torch.LongTensor(np.array(self.actions)).to(device),
            "rewards": torch.FloatTensor(np.array(self.rewards)).to(device),
            "values": torch.FloatTensor(np.array(self.values)).to(device),
            "log_probs": torch.FloatTensor(np.array(self.log_probs)).to(device),
            "dones": torch.FloatTensor(np.array(self.dones)).to(device),
        }

    def __len__(self):
        return len(self.states)
