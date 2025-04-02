# src/deeprl/runners/ppo_runner.py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ..losses import compute_ppo_loss, compute_value_loss, compute_entropy_bonus
from ..memory import RolloutBuffer
from .base_runner import BaseRunner

class PPORunner(BaseRunner):
    """
    Runner for Proximal Policy Optimization.
    """
    def __init__(self, env, device, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        """
        Initialize the PPO runner.
        
        Args:
            env (gym.Env): Gym environment
            device (torch.device): Device to use for tensor operations
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            clip_epsilon (float): PPO clipping parameter
            value_coef (float): Value loss coefficient
            entropy_coef (float): Entropy bonus coefficient
            max_grad_norm (float): Maximum gradient norm
        """
        super(PPORunner, self).__init__(env, device, gamma)
        
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()
        
        # Performance tracking
        self.consecutive_worse_iterations = 0
    def collect_rollouts(self, model, n_steps=2048):
        """
        Collect experience using the current policy.
        
        Args:
            model (BaseModel): Model to use for action selection
            n_steps (int): Number of steps to collect
            
        Returns:
            RolloutBuffer: Buffer with collected experience
        """
        # Clear rollout buffer
        self.rollout_buffer.clear()
        
        # Reset environment
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        
        for _ in range(n_steps):
            # Get action from model
            action, log_prob, value = model.get_action(state, self.device)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.rollout_buffer.add(state, action, reward, value, log_prob, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # If episode ended, reset environment and track reward
            if done:
                self.episode_rewards.append(episode_reward)
                self.reward_window.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0
        
        # Get final value for bootstrapping
        if not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, last_value = model(state_tensor)
                last_value = last_value.item()
        else:
            last_value = 0
        
        # Compute returns and advantages
        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda)
        
        return self.rollout_buffer, returns, advantages
    
    # src/deeprl/runners/ppo_runner.py (update_policy method)
    def update_policy(self, model, optimizer, rollout_buffer, returns, advantages, batch_size=64, epochs=10):
        """
        Update the policy using PPO.
        
        Args:
            model (BaseModel): Model to update
            optimizer (torch.optim.Optimizer): Optimizer to use
            rollout_buffer (RolloutBuffer): Buffer with collected experience
            returns (numpy.ndarray): Computed returns
            advantages (numpy.ndarray): Computed advantages
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train
        """
        # Get data from buffer
        buffer_data = rollout_buffer.get_data(self.device)
        
        # Normalize advantages
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Convert returns to tensor
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            buffer_data['states'],
            buffer_data['actions'],
            returns_tensor,
            advantages_tensor,
            buffer_data['log_probs']
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train model
        model.train()
        for epoch in range(epochs):
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            for batch_states, batch_actions, batch_returns, batch_advantages, batch_old_log_probs in dataloader:
                # Forward pass
                new_log_probs, values, entropy, _ = model.evaluate_actions(batch_states, batch_actions)
                
                # Compute losses
                policy_loss = compute_ppo_loss(
                    new_log_probs, batch_old_log_probs, batch_advantages, self.clip_epsilon)
                value_loss = compute_value_loss(values, batch_returns)
                entropy_bonus = compute_entropy_bonus(entropy)  # Now entropy is a tensor
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Policy Loss: {total_policy_loss/len(dataloader):.4f}, "
                    f"Value Loss: {total_value_loss/len(dataloader):.4f}, "
                    f"Entropy: {total_entropy/len(dataloader):.4f}")
    
    def check_performance(self, threshold=5):
        """
        Check if performance has degraded and human intervention is needed.
        
        Args:
            threshold (int): Number of consecutive worse iterations before intervention
            
        Returns:
            bool: Whether human intervention is needed
        """
        # Need at least a few episodes to measure performance
        if len(self.reward_window) < 5:
            return False
        
        current_mean_reward = np.mean(list(self.reward_window))
        
        # If current performance is better than best, update best
        if current_mean_reward > self.best_reward:
            self.best_reward = current_mean_reward
            self.consecutive_worse_iterations = 0
            return False
        
        # If performance is significantly worse than best
        if current_mean_reward < 0.7 * self.best_reward:
            self.consecutive_worse_iterations += 1
        else:
            # Reset counter if not too bad
            self.consecutive_worse_iterations = 0
        
        # If performance has been consistently worse for threshold iterations
        if self.consecutive_worse_iterations >= threshold:
            return True
        
        return False
