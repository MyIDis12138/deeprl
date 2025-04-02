# src/deeprl/models/actor_critic.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..layers import CNNFeatureExtractor
from .base_model import BaseModel


class ActorCritic(BaseModel):
    """
    Actor-Critic model for discrete action spaces.
    """

    def __init__(self, input_shape, n_actions):
        """
        Initialize the Actor-Critic model.

        Args:
            input_shape (tuple): Shape of input (channels, height, width)
            n_actions (int): Number of discrete actions
        """
        super(ActorCritic, self).__init__()

        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor(input_shape)
        feature_size = self.feature_extractor.feature_size

        # Actor (policy) network
        self.actor = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

        # Critic (value) network
        self.critic = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        """
        Forward pass through the Actor-Critic model.

        Args:
            x (torch.Tensor): Input tensor [batch, height, width, channels]

        Returns:
            tuple: (action_logits, state_values)
        """
        # Normalize input
        x = x / 255.0

        # Permute dimensions for CNN [batch, height, width, channels] -> [batch, channels, height, width]
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)

        # Extract features
        features = self.feature_extractor(x)

        # Get action logits and state values
        action_logits = self.actor(features)
        state_values = self.critic(features)

        return action_logits, state_values

    def get_action(self, state, device, deterministic=False):
        """
        Get an action from the policy.

        Args:
            state (numpy.ndarray): State observation
            device (torch.device): Device to perform computation on
            deterministic (bool): Whether to use deterministic action selection

        Returns:
            tuple: (action, log_prob, value)
        """
        # Convert state to tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        elif state.dim() == 3:
            state = state.unsqueeze(0).to(device)

        with torch.no_grad():
            action_logits, state_value = self.forward(state)

            if deterministic:
                # Use argmax for deterministic behavior
                action = torch.argmax(action_logits, dim=-1)
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                log_prob = dist.log_prob(action)
            else:
                # Sample from distribution for stochastic behavior
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def evaluate_actions(self, states, actions):
        """
        Evaluate actions by computing log probs, values, and entropy.

        Args:
            states (torch.Tensor): Batch of states
            actions (torch.Tensor): Batch of actions

        Returns:
            tuple: (log_probs, values, entropy, action_probs)
        """
        action_logits, values = self.forward(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy, action_probs
