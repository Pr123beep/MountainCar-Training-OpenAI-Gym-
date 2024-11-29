
# MountainCar Reinforcement Learning

## Overview

This repository contains a reinforcement learning implementation for the classic `MountainCar-v0` environment from OpenAI Gym. The project uses Q-learning to train an agent to solve the environment efficiently with custom reward modifications for enhanced learning.

The repository includes:
1. **`mountainCar.py`**: Implements a custom MountainCar environment with modified rewards to encourage faster learning and success.
2. **`renderMountain.py`**: Demonstrates how to load a pre-trained Q-table and visualize the agent's performance in the environment.

## Features
- **Custom Reward Function**: Enhances the default reward system to guide the agent effectively.
- **Q-Learning**: Implements a tabular Q-learning approach with discretized state spaces.
- **Visualization**: Includes rendering capabilities for a human-readable view of the agent's performance.
- **Pre-trained Q-Table**: Load and test a pre-trained Q-table for immediate results.

## Requirements
Ensure you have the following Python libraries installed:
- `gym`
- `numpy`
- `pygame`
- `pickle`

Install them using:
```bash
pip install gym pygame numpy
```

## File Details

### `mountainCar.py`
- **Purpose**: Provides the training setup for the MountainCar environment using Q-learning.
- **Key Features**:
  - Custom reward shaping to promote faster convergence.
  - Environment wrapping to incorporate additional logic during training.
  - Training the agent and saving the Q-table.

### `renderMountain.py`
- **Purpose**: Loads a pre-trained Q-table and visualizes the agent's performance.
- **Key Features**:
  - Discretizes continuous states into discrete bins for efficient Q-table indexing.
  - Renders the agent's progress using OpenAI Gym's visualization tools.

## How to Use

### Train the Agent
1. Open `mountainCar.py`.
2. Run the script to train the agent. The Q-table will be saved as `q_table.pkl`.

```bash
python mountainCar.py
```

### Visualize the Trained Agent
1. Ensure `q_table.pkl` is present in the directory.
2. Run `renderMountain.py` to load the Q-table and render the agent's performance.

```bash
python renderMountain.py
```

### Customization
You can modify the following parameters in `mountainCar.py` to experiment with different setups:
- **State Discretization**: Adjust `num_bins` for finer or coarser discretization.
- **Reward Function**: Modify the reward logic in the `CustomMountainCarEnv` class.
- **Training Episodes**: Increase or decrease the number of episodes for training.

## Results
The custom reward system enables the agent to learn faster and reach the goal state effectively. The pre-trained Q-table provided offers a quick demonstration of this capability.

## License
This project is licensed under the MIT License. Feel free to use and modify the code for your own purposes.

## Contact
For any queries or suggestions, feel free to reach out.
