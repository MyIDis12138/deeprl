# DeepRL

A modular reinforcement learning library for learning and experimenting with state-of-the-art RL algorithms.

## Features

- Modular design with clean abstractions for environments, models, and training algorithms
- Support for Gymnasium environments (formerly OpenAI Gym)
- Implementation of PPO (Proximal Policy Optimization) algorithm
- Human intervention capability for guided learning
- YAML-based configuration system for easy experimentation

## Getting Started

### Installation

1. Clone the repository
```bash
git clone https://github.com/yanggu123138/deeprl.git
cd deeprl
```

2. Create a virtual environment and install dependencies
```bash
uv .venv python 3.12
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
uv pip install -e .[dev]
```

### Configuration

DeepRL uses YAML files for configuration. The default configuration is located at `config/default.yaml`. You can create your own configuration files based on this template.

#### Configuration Structure

The configuration is organized in the following sections:

- `env`: Environment settings (ID, render mode, etc.)
- `model`: Model architecture settings
- `training`: Training hyperparameters
- `paths`: Paths for saving models, logs, etc.


### Running

To train an agent using the default configuration:

```bash
python src/main.py
```

To use a custom configuration file:

```bash
python src/main.py --config path/to/your/config.yaml
```

To evaluate a trained model:

```bash
python src/main.py --eval --model-path models/ppo_final.pt
```
