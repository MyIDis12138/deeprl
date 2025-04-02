# src/deeprl/losses/__init__.py
from .ppo_losses import compute_ppo_loss, compute_value_loss, compute_entropy_bonus

__all__ = ['compute_ppo_loss', 'compute_value_loss', 'compute_entropy_bonus']