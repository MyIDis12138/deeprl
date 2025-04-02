import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Neural Network Architecture
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()

        # Feature extraction layers (shared)
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate feature size after convolutions
        feature_size = self._get_conv_output(input_shape)

        # Actor (policy) network
        self.actor = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

        # Critic (value) network
        self.critic = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def _get_conv_output(self, shape):
        # Helper function to calculate conv output size
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Normalize input
        x = x / 255.0

        # x shape: [batch, height, width, channels] -> [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)

        features = self.features(x)
        action_logits = self.actor(features)
        state_values = self.critic(features)

        return action_logits, state_values

    def get_action(self, state, deterministic=False):
        # Convert state to tensor, add batch dimension if not present
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        elif state.dim() == 3:
            state = state.unsqueeze(0).to(device)

        with torch.no_grad():
            action_logits, state_value = self.forward(state)

            if deterministic:
                # Use argmax for deterministic behavior
                action = torch.argmax(action_logits, dim=-1)
            else:
                # Sample from distribution for stochastic behavior
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                action = dist.sample()

        return action.item(), state_value.item()


# PPO Agent with Adaptive Human Intervention
class AdaptivePPOAgent:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        epochs=10,
    ):

        self.env = env
        self.obs_shape = (3, 96, 96)  # Channels first for PyTorch
        self.n_actions = env.action_space.n

        # Initialize actor-critic network
        self.policy = ActorCritic(self.obs_shape, self.n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs

        # Training metrics
        self.episode_rewards = []
        self.human_intervention_points = []

        # Performance tracking
        self.reward_window = deque(maxlen=10)
        self.best_reward = -float("inf")
        self.consecutive_worse_iterations = 0

        # Human demonstration buffer
        self.demo_buffer = []

    def collect_human_demonstrations(self, n_episodes=3, is_intervention=False):
        """Collect human demonstrations"""
        if is_intervention:
            print("\n===== HUMAN INTERVENTION REQUIRED =====")
            print("Performance has dropped for several iterations.")
            print("Please take control to demonstrate good driving again.")
        else:
            print("\n===== STARTING HUMAN DEMONSTRATION PHASE =====")

        print(f"You will control the car for {n_episodes} episodes.")
        print("Controls: Arrow keys for steering and acceleration")
        print("- LEFT/RIGHT: Steer left/right")
        print("- UP: Accelerate")
        print("- DOWN: Brake")
        print("- Press SPACE to pause/unpause")

        # Initialize pygame for key capture
        pygame.init()

        demo_states = []
        demo_actions = []
        demo_rewards = []
        demo_next_states = []
        demo_dones = []

        # Track when this intervention happened
        if is_intervention:
            self.human_intervention_points.append(len(self.episode_rewards))

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            paused = False

            while not done:
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
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
                demo_states.append(state)
                demo_actions.append(action)
                demo_rewards.append(reward)
                demo_next_states.append(next_state)
                demo_dones.append(done)

                # Update state
                state = next_state
                episode_reward += reward
                step += 1

                # Control display speed
                time.sleep(0.01)

            print(f"Episode {episode + 1}/{n_episodes} - Reward: {episode_reward:.2f}, Steps: {step}")
            self.episode_rewards.append(episode_reward)
            self.reward_window.append(episode_reward)

        # Store demonstration data
        self.demo_buffer = {
            "states": np.array(demo_states),
            "actions": np.array(demo_actions),
            "rewards": np.array(demo_rewards),
            "next_states": np.array(demo_next_states),
            "dones": np.array(demo_dones),
        }

        # Update best reward if needed
        current_mean_reward = np.mean(list(self.reward_window))
        if current_mean_reward > self.best_reward:
            self.best_reward = current_mean_reward

        # Reset consecutive worse iterations counter after human intervention
        self.consecutive_worse_iterations = 0

        if is_intervention:
            print("\n===== HUMAN INTERVENTION COMPLETED =====")
        else:
            print("\n===== HUMAN DEMONSTRATION PHASE COMPLETED =====")

        return self.demo_buffer

    def learn_from_demonstrations(self):
        """Use collected demonstrations to update policy"""
        if not self.demo_buffer:
            print("No demonstrations available.")
            return

        print("Learning from human demonstrations...")

        # Convert demo data
        states = torch.FloatTensor(self.demo_buffer["states"]).to(device)
        actions = torch.LongTensor(self.demo_buffer["actions"]).to(device)

        # Calculate returns from rewards
        returns = []
        next_return = 0
        for reward, done in zip(reversed(self.demo_buffer["rewards"]), reversed(self.demo_buffer["dones"])):
            if done:
                next_return = 0
            next_return = reward + self.gamma * next_return
            returns.insert(0, next_return)
        returns = torch.FloatTensor(returns).to(device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(states, actions, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Update policy using behavior cloning and value learning
        for _ in range(self.epochs * 3):  # More epochs for demonstrations
            for batch_states, batch_actions, batch_returns in dataloader:
                # Forward pass
                action_logits, state_values = self.policy(batch_states)

                # Get action probabilities
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)

                # Calculate log probabilities of actions
                log_probs = dist.log_prob(batch_actions)

                # Behavior cloning loss (negative log likelihood)
                actor_loss = -log_probs.mean()

                # Value function loss
                value_loss = 0.5 * ((state_values.squeeze() - batch_returns) ** 2).mean()

                # Total loss
                loss = actor_loss + self.value_coef * value_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        print("Finished learning from demonstrations.")

    def collect_rollouts(self, n_steps=2048):
        """Collect experience using current policy"""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        # Reset environment
        state, _ = self.env.reset()
        done = False
        episode_reward = 0

        for _ in range(n_steps):
            # Get action from policy
            action, value = self.policy.get_action(state)

            # Calculate log probability
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_logits, _ = self.policy(state_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor(action, device=device))

            # Take action in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob.item())
            dones.append(done)

            episode_reward += reward

            # Update state
            state = next_state

            # If episode ended, reset environment and track reward
            if done:
                self.episode_rewards.append(episode_reward)
                self.reward_window.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0

        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        values = np.array(values)
        log_probs = np.array(log_probs)
        dones = np.array(dones)

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = np.zeros_like(rewards)
        last_gae = 0

        # Get value of the last state if episode is not done
        with torch.no_grad():
            if not done:
                last_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                _, last_value = self.policy(last_state)
                last_value = last_value.item()
            else:
                last_value = 0

        # Calculate advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - done
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Calculate returns
        returns = advantages + values

        return states, actions, returns, advantages, log_probs

    def update_policy(self, states, actions, returns, advantages, old_log_probs):
        """Update policy using the collected experience"""
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(states, actions, returns, advantages, old_log_probs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PPO update
        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_returns, batch_advantages, batch_old_log_probs in dataloader:
                # Forward pass
                action_logits, state_values = self.policy(batch_states)

                # Get action probabilities
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)

                # Calculate log probabilities of actions
                new_log_probs = dist.log_prob(batch_actions)

                # Calculate entropy
                entropy = dist.entropy().mean()

                # Calculate ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value function loss
                value_loss = 0.5 * ((state_values.squeeze() - batch_returns) ** 2).mean()

                # Total loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def check_performance(self, threshold=5):
        """Check if performance has degraded and human intervention is needed"""
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

    def train_adaptive(
        self, n_iterations=50, n_steps_per_iteration=2048, initial_human_episodes=5, intervention_episodes=3, intervention_threshold=5
    ):
        """
        Train with adaptive human intervention

        Args:
            n_iterations: Total number of PPO iterations
            n_steps_per_iteration: Number of environment steps per iteration
            initial_human_episodes: Number of initial human demonstration episodes
            intervention_episodes: Number of episodes per human intervention
            intervention_threshold: Number of consecutive worse iterations before intervention
        """
        # Phase 1: Initial human demonstration
        self.collect_human_demonstrations(n_episodes=initial_human_episodes)

        # Phase 2: Learn from demonstrations
        self.learn_from_demonstrations()

        # Phase 3: PPO training with adaptive human intervention
        print(f"\nStarting adaptive PPO training for {n_iterations} iterations...")

        for i in range(n_iterations):
            # Collect experience
            states, actions, returns, advantages, old_log_probs = self.collect_rollouts(n_steps=n_steps_per_iteration)

            # Update policy
            self.update_policy(states, actions, returns, advantages, old_log_probs)

            # Print progress
            if len(self.reward_window) > 0:
                mean_reward = np.mean(list(self.reward_window))
                print(
                    f"Iteration {i + 1}/{n_iterations}, \
                      Mean Reward (window): {mean_reward:.2f}, \
                      Best: {self.best_reward:.2f}, \
                      Worse Count: {self.consecutive_worse_iterations}"
                )

            # Check if performance has degraded
            if self.check_performance(threshold=intervention_threshold):
                print(f"\nPerformance has degraded for {intervention_threshold} consecutive iterations.")
                # Phase 4: Human intervention
                self.collect_human_demonstrations(n_episodes=intervention_episodes, is_intervention=True)
                # Learn from new demonstrations
                self.learn_from_demonstrations()

    def save(self, path):
        """Save the model"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_reward": self.best_reward,
                "episode_rewards": self.episode_rewards,
                "human_intervention_points": self.human_intervention_points,
            },
            path,
        )

    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_reward = checkpoint.get("best_reward", -float("inf"))
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.human_intervention_points = checkpoint.get("human_intervention_points", [])

    def plot_rewards(self):
        """Plot the rewards over training with intervention markers"""
        plt.figure(figsize=(12, 6))

        # Plot all rewards
        plt.plot(self.episode_rewards, "b-", alpha=0.7)

        # Mark initial human demonstration episodes
        init_human_end = min(len(self.episode_rewards), 5)
        plt.axvspan(0, init_human_end - 1, color="g", alpha=0.3, label="Initial Human Demo")

        # Mark human intervention points
        for point in self.human_intervention_points:
            plt.axvline(x=point, color="r", linestyle="--")
            plt.axvspan(point, point + 2, color="r", alpha=0.3)

        # Add a smoother trend line
        if len(self.episode_rewards) > 10:
            window_size = min(10, len(self.episode_rewards) // 5)
            smoothed = np.convolve(self.episode_rewards, np.ones(window_size) / window_size, mode="valid")
            plt.plot(range(window_size - 1, len(self.episode_rewards)), smoothed, "k-", linewidth=2, label="Trend")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards with Human Interventions")

        # Add legend with custom entries
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="g", alpha=0.3, label="Initial Human Demo"),
            Patch(facecolor="r", alpha=0.3, label="Human Intervention"),
            Patch(facecolor="b", alpha=0.7, label="Agent Episodes"),
        ]
        if len(self.episode_rewards) > 10:
            legend_elements.append(Patch(color="k", label="Trend"))

        plt.legend(handles=legend_elements)
        plt.grid(True, alpha=0.3)
        plt.savefig("adaptive_ppo_rewards.png")
        plt.show()


def visualize_agent(agent, n_episodes=3, fps=30, deterministic=True):
    """Visualize the trained agent"""
    env = agent.env

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        # Create figure to display observation
        plt.figure(figsize=(5, 5))
        img_plot = plt.imshow(state)
        plt.title("Agent's Observation")
        plt.draw()
        plt.pause(0.001)

        while not done:
            # Get action from policy
            action, _ = agent.policy.get_action(state, deterministic=deterministic)

            # Display agent's perspective
            img_plot.set_data(state)
            plt.draw()
            plt.pause(0.001)

            # Take action in environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step += 1

            # Control visualization speed
            time.sleep(1 / fps)

        print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f} in {step} steps")

    plt.close("all")


# Main training function
def main():
    # Create environment
    env = gym.make("CarRacing-v3", render_mode="human", continuous=False)

    # Create adaptive agent
    agent = AdaptivePPOAgent(env)

    # Train agent with adaptive human intervention
    agent.train_adaptive(
        n_iterations=30,  # Total PPO iterations
        initial_human_episodes=5,  # Initial human demos
        intervention_episodes=3,  # Episodes per intervention
        intervention_threshold=5,  # How many bad iterations before intervention
    )

    # Save the trained model
    agent.save("adaptive_ppo_car_racing.pt")

    # Plot rewards
    agent.plot_rewards()

    # Visualize final agent
    print("Visualizing trained agent...")
    visualize_agent(agent, n_episodes=3)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
