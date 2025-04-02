# src/main.py (updated for YAML config)
import argparse
import os

import torch
import torch.optim as optim

from deeprl.config import load_config
from deeprl.gym_envs import build_env
from deeprl.model import ActorCritic
from deeprl.runners import AdaptiveCarRacingRunner


def main():
    parser = argparse.ArgumentParser(description="Adaptive PPO with Human Intervention for CarRacing")
    parser.add_argument("config", type=str, default="config/default.yaml", help="Path to YAML configuration file")
    parser.add_argument("--eval", action="store_true", help="Evaluate model instead of training")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model for evaluation")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seeds
    torch.manual_seed(config["training"]["random_seed"])
    torch.cuda.manual_seed(config["training"]["random_seed"])

    # Create directories
    os.makedirs(config["paths"]["model_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment using config
    env = build_env(config["env"]["id"], render_mode=config["env"]["render_mode"], continuous=config["env"]["continuous"])

    # Parse input shape from config
    input_shape = tuple(config["model"]["input_shape"])

    # Create model
    n_actions = env.action_space.n
    model = ActorCritic(input_shape, n_actions, hidden_size=config["model"]["hidden_size"]).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Create runner with config parameters
    car_racing_runner = AdaptiveCarRacingRunner(
        env,
        device,
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
        clip_epsilon=config["training"]["clip_epsilon"],
        value_coef=config["training"]["value_coef"],
        entropy_coef=config["training"]["entropy_coef"],
        max_grad_norm=config["training"]["max_grad_norm"],
    )

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

    # Training mode - convert config to an object that mimics args
    class ConfigArgs:
        pass

    config_args = ConfigArgs()
    config_args.env_id = config["env"]["id"]
    config_args.iterations = config["training"]["iterations"]
    config_args.steps = config["training"]["steps_per_iteration"]
    config_args.batch_size = config["training"]["batch_size"]
    config_args.threshold = config["training"]["intervention_threshold"]

    car_racing_runner.train(model, optimizer, config_args)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
