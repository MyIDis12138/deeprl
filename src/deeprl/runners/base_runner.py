# src/deeprl/runners/base_runner.py
import torch
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

class BaseRunner:
    """
    Base class for all runners.
    """
    def __init__(self, env, device, gamma=0.99):
        """
        Initialize the base runner.
        
        Args:
            env (gym.Env): Gym environment
            device (torch.device): Device to use for tensor operations
            gamma (float): Discount factor
        """
        self.env = env
        self.device = device
        self.gamma = gamma
        
        # Performance tracking
        self.episode_rewards = []
        self.reward_window = deque(maxlen=10)
        self.best_reward = -float('inf')
    
    def evaluate(self, model, n_episodes=3, render=True, deterministic=True):
        """
        Evaluate a model.
        
        Args:
            model (BaseModel): Model to evaluate
            n_episodes (int): Number of episodes to evaluate
            render (bool): Whether to render the environment
            deterministic (bool): Whether to use deterministic action selection
            
        Returns:
            float: Mean reward
        """
        rewards = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get action from model
                action, _, _ = model.get_action(state, self.device, deterministic=deterministic)
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Render if specified
                if render:
                    self.env.render()
                    time.sleep(0.01)
            
            rewards.append(episode_reward)
            print(f"Episode {episode+1} reward: {episode_reward:.2f}")
        
        mean_reward = np.mean(rewards)
        print(f"Mean reward over {n_episodes} episodes: {mean_reward:.2f}")
        
        return mean_reward
    
    def plot_rewards(self, save_path=None, window_size=10):
        """
        Plot episode rewards.
        
        Args:
            save_path (str): Path to save the plot
            window_size (int): Window size for smoothing
        """
        plt.figure(figsize=(10, 6))
        
        # Plot episode rewards
        plt.plot(self.episode_rewards, 'b-', alpha=0.3)
        
        # Plot smoothed rewards
        if len(self.episode_rewards) >= window_size:
            smoothed = []
            for i in range(len(self.episode_rewards) - window_size + 1):
                smoothed.append(np.mean(self.episode_rewards[i:i+window_size]))
            plt.plot(range(window_size-1, len(self.episode_rewards)), smoothed, 'r-')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

