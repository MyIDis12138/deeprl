# src/deeprl/losses/ppo_losses.py
import torch


def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=0.2):
    """
    Compute the PPO clipped surrogate loss.

    Args:
        new_log_probs (torch.Tensor): Log probabilities of actions under current policy
        old_log_probs (torch.Tensor): Log probabilities of actions under old policy
        advantages (torch.Tensor): Advantage estimates
        clip_epsilon (float): Clipping parameter

    Returns:
        torch.Tensor: PPO loss
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    return -torch.min(surrogate1, surrogate2).mean()


def compute_value_loss(values, returns):
    """
    Compute the value function loss.

    Args:
        values (torch.Tensor): Predicted values
        returns (torch.Tensor): Target returns

    Returns:
        torch.Tensor: Value loss
    """
    return 0.5 * ((values.squeeze() - returns) ** 2).mean()


def compute_entropy_bonus(entropy):
    """
    Compute the entropy bonus for exploration.

    Args:
        entropy (torch.Tensor): Entropy tensor

    Returns:
        torch.Tensor: Mean entropy
    """
    # Just return the mean of the entropy tensor
    return entropy.mean()
