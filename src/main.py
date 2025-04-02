# src/main.py (updated)
import argparse
import os

import torch
import torch.optim as optim

from deeprl.gym_envs import build_env
from deeprl.model import ActorCritic
from deeprl.runners import AdaptiveCarRacingRunner


def main():
    parser = argparse.ArgumentParser(description="Adaptive PPO with Human Intervention for CarRacing")
    parser.add_argument("--env-id", type=str, default="CarRacing-v3", help="Gym environment ID")
    parser.add_argument("--iterations", type=int, default=30, help="Number of PPO iterations")
    parser.add_argument("--steps", type=int, default=2048, help="Steps per iteration")
    parser.add_argument("--init-human", type=int, default=0, help="Initial human demonstration episodes")
    parser.add_argument("--intervention", type=int, default=3, help="Episodes per intervention")
    parser.add_argument("--threshold", type=int, default=5, help="Threshold for intervention")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Evaluate model instead of training")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model for evaluation")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment using your builder
    env = build_env(args.env_id, render_mode="human", continuous=False)

    # Create model
    input_shape = (3, 96, 96)  # Channels first for CNN
    n_actions = env.action_space.n
    model = ActorCritic(input_shape, n_actions).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create car racing runner
    car_racing_runner = AdaptiveCarRacingRunner(env, device, gamma=args.gamma)

    if args.eval:
        # Evaluation mode
        if args.model_path is None:
            print("Please provide a model path for evaluation")
            return

        print(f"Loading model from {args.model_path}")
        model.load(args.model_path, device)
        print("Evaluating model...")
        car_racing_runner.evaluate(model, n_episodes=5)
        return

    # Training mode
    car_racing_runner.train(model, optimizer, args)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
