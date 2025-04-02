# src/deeprl/runners/car_racing_runner.py
import numpy as np
import torch
import pygame
import time
import matplotlib.pyplot as plt
from collections import deque

from .human_interaction import HumanInteractionRunner
from .ppo_runner import PPORunner

class AdaptiveCarRacingRunner:
    """
    Specialized runner for the CarRacing environment with adaptive human intervention.
    """
    def __init__(self, env, device, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        """
        Initialize the adaptive runner.
        
        Args:
            env: Gym environment
            device: Device to use for tensor operations
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
        """
        self.env = env
        self.device = device
        
        # Create specialized runners
        self.human_runner = HumanInteractionRunner(env, device, gamma=gamma)
        self.ppo_runner = PPORunner(env, device, gamma=gamma, gae_lambda=gae_lambda,
                                    clip_epsilon=clip_epsilon, value_coef=value_coef,
                                    entropy_coef=entropy_coef, max_grad_norm=max_grad_norm)
    
    def train(self, model, optimizer, args):
        """
        Train the model with adaptive human intervention.
        
        Args:
            model: Model to train
            optimizer: Optimizer to use
            args: Training arguments
        """
        print(f"Starting adaptive PPO training with human intervention on {args.env_id}")
        
        # Phase 1: Initial human demonstration
        print("\n==== Phase 1: Initial Human Demonstration ====")
        demo_buffer = self.human_runner.collect_demonstrations(n_episodes=args.init_human)
        
        # Phase 2: Learn from demonstrations
        print("\n==== Phase 2: Learning from Demonstrations ====")
        self.human_runner.learn_from_demonstrations(model, optimizer, batch_size=args.batch_size)
        
        # Phase 3: PPO training with adaptive human intervention
        print("\n==== Phase 3: Adaptive PPO Training ====")
        
        for iteration in range(args.iterations):
            print(f"\nIteration {iteration+1}/{args.iterations}")
            
            # Collect rollouts
            rollout_buffer, returns, advantages = self.ppo_runner.collect_rollouts(model, n_steps=args.steps)
            
            # Update policy
            self.ppo_runner.update_policy(
                model, optimizer, rollout_buffer, returns, advantages, 
                batch_size=args.batch_size
            )
            
            # Copy episode rewards for plotting
            self.human_runner.episode_rewards = self.ppo_runner.episode_rewards.copy()
            
            # Print current performance
            mean_reward = np.mean(list(self.ppo_runner.reward_window)) if len(self.ppo_runner.reward_window) > 0 else 0
            print(f"Mean reward: {mean_reward:.2f}, Best: {self.ppo_runner.best_reward:.2f}, "
                  f"Worse count: {self.ppo_runner.consecutive_worse_iterations}")
            
            # Save checkpoint
            model_path = f"models/ppo_checkpoint_{iteration+1}.pt"
            model.save(model_path)
            print(f"Saved checkpoint to {model_path}")
            
            # Check if performance has degraded
            if self.ppo_runner.check_performance(threshold=args.threshold):
                print("\n==== Performance Degraded: Requesting Human Intervention ====")
                
                # Human intervention
                demo_buffer = self.human_runner.collect_demonstrations(n_episodes=args.intervention, is_intervention=True)
                
                # Learn from new demonstrations
                self.human_runner.learn_from_demonstrations(model, optimizer, batch_size=args.batch_size)
                
                # Reset consecutive worse iterations counter
                self.ppo_runner.consecutive_worse_iterations = 0
        
        # Save final model
        final_model_path = "models/ppo_final.pt"
        model.save(final_model_path)
        print(f"\nTraining completed. Final model saved to {final_model_path}")
        
        # Plot rewards
        self.human_runner.plot_rewards_with_interventions(save_path="rewards_with_interventions.png")
        
        # Final evaluation
        print("\n==== Final Evaluation ====")
        self.ppo_runner.evaluate(model, n_episodes=3, render=True, deterministic=True)
    
    def evaluate(self, model, n_episodes=5):
        """
        Evaluate the model.
        
        Args:
            model: Model to evaluate
            n_episodes: Number of episodes to evaluate
        """
        return self.ppo_runner.evaluate(model, n_episodes=n_episodes, render=True, deterministic=True)

