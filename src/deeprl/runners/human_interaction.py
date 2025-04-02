# src/deeprl/runners/human_interaction.py
import time

import numpy as np
import pygame
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..memory import HumanDemonstrationBuffer
from .base_runner import BaseRunner


class HumanInteractionRunner(BaseRunner):
    """
    Runner for human interaction and demonstration collection.
    """

    def __init__(self, env, device, gamma=0.99):
        """
        Initialize the human interaction runner.

        Args:
            env (gym.Env): Gym environment
            device (torch.device): Device to use for tensor operations
            gamma (float): Discount factor
        """
        super(HumanInteractionRunner, self).__init__(env, device, gamma)

        # Initialize pygame for key capture
        pygame.init()

        # Mapping from pygame keys to CarRacing actions
        self.key_to_action = {
            pygame.K_LEFT: 1,  # Left arrow -> steer left
            pygame.K_RIGHT: 2,  # Right arrow -> steer right
            pygame.K_UP: 3,  # Up arrow -> accelerate
            pygame.K_DOWN: 4,  # Down arrow -> brake
        }

        # Demonstration buffer
        self.demo_buffer = HumanDemonstrationBuffer()

        # Human intervention tracking
        self.human_intervention_points = []

    def collect_demonstrations(self, n_episodes=5, is_intervention=False):
        """
        Collect human demonstrations.

        Args:
            n_episodes (int): Number of episodes to collect
            is_intervention (bool): Whether this is an intervention during training

        Returns:
            HumanDemonstrationBuffer: Buffer with collected demonstrations
        """
        if is_intervention:
            print("\n===== HUMAN INTERVENTION REQUIRED =====")
            print("Performance has dropped. Please take control to demonstrate good driving.")
            # Track when this intervention happened
            self.human_intervention_points.append(len(self.episode_rewards))
        else:
            print("\n===== STARTING HUMAN DEMONSTRATION PHASE =====")

        print(f"You will control the car for {n_episodes} episodes.")
        print("Controls:")
        print("- LEFT/RIGHT: Steer left/right")
        print("- UP: Accelerate")
        print("- DOWN: Brake")
        print("- SPACE: Pause/unpause")
        print("- ESC: Quit")

        # Clear the demonstration buffer
        self.demo_buffer.clear()

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            paused = False

            while not done:
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        pygame.quit()
                        return self.demo_buffer
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        paused = not paused
                        if paused:
                            print("Game paused. Press SPACE to continue.")
                        else:
                            print("Game resumed.")

                if paused:
                    time.sleep(0.1)
                    continue

                # Display current state
                self.env.render()

                # Get keyboard input
                keys = pygame.key.get_pressed()

                # Default action is do nothing (0)
                action = 0

                # Check which keys are pressed
                if keys[pygame.K_LEFT]:
                    action = 1
                elif keys[pygame.K_RIGHT]:
                    action = 2
                elif keys[pygame.K_UP]:
                    action = 3
                elif keys[pygame.K_DOWN]:
                    action = 4

                # Take action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store experience
                self.demo_buffer.add(state, action, reward, next_state, done)

                # Update state
                state = next_state
                episode_reward += reward
                step += 1

                # Control display speed
                time.sleep(0.01)

            print(f"Episode {episode + 1}/{n_episodes} - Reward: {episode_reward:.2f}, Steps: {step}")
            self.episode_rewards.append(episode_reward)
            self.reward_window.append(episode_reward)

            # Update best reward if needed
            current_mean_reward = np.mean(list(self.reward_window))
            if current_mean_reward > self.best_reward:
                self.best_reward = current_mean_reward

        if is_intervention:
            print("\n===== HUMAN INTERVENTION COMPLETED =====")
        else:
            print("\n===== HUMAN DEMONSTRATION PHASE COMPLETED =====")

        return self.demo_buffer

    def learn_from_demonstrations(self, model, optimizer, batch_size=64, epochs=30):
        """
        Train a model using human demonstrations with behavior cloning.

        Args:
            model (BaseModel): Model to train
            optimizer (torch.optim.Optimizer): Optimizer to use
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train
        """
        if len(self.demo_buffer) == 0:
            print("No demonstrations available for learning.")
            return

        print(f"Learning from {len(self.demo_buffer)} demonstration steps...")

        # Get data from buffer
        demo_data = self.demo_buffer.get_data(self.device)
        returns = torch.FloatTensor(self.demo_buffer.compute_returns(self.gamma)).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(demo_data["states"], demo_data["actions"], returns)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train model
        model.train()
        for epoch in range(epochs):
            total_actor_loss = 0
            total_value_loss = 0

            for batch_states, batch_actions, batch_returns in dataloader:
                # Forward pass
                log_probs, values, _, _ = model.evaluate_actions(batch_states, batch_actions)

                # Compute losses
                actor_loss = -log_probs.mean()  # Behavior cloning (negative log likelihood)
                value_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()

                # Total loss
                loss = actor_loss + 0.5 * value_loss

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, \
                      Actor Loss: {total_actor_loss / len(dataloader):.4f}, \
                      Value Loss: {total_value_loss / len(dataloader):.4f}"
                )

        print("Finished learning from demonstrations.")
