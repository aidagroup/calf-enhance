# Underwater Drone Environment

A Gymnasium environment for an underwater drone simulation with Pygame rendering.

## Description

The Underwater Drone Environment simulates a drone in an underwater environment that must navigate toward a surface opening. The drone has control over longitudinal (forward/backward) and lateral (left/right) thrust.

Key features:
- Realistic physics with water drag
- Momentum-based movement
- Pygame visualization
- Goal-oriented task (navigate to the surface opening)
- Compatible with the Gymnasium API

In this branch we will enhance the animations.

## Installation

```bash
# Install dependencies
pip install numpy gymnasium pygame
```

## Usage

```python
import gymnasium as gym
from src.envs import UnderwaterDroneEnv

# Create environment
env = UnderwaterDroneEnv(render_mode="human")

# Reset environment
observation, info = env.reset()

# Step through environment
for _ in range(1000):
    action = env.action_space.sample()  # Or your policy
    observation, reward, terminated, truncated, info = env.step(action)
    
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

## Example

An example script is provided in `examples/underwater_drone_demo.py`:

```bash
python examples/underwater_drone_demo.py
```

## Environment Details

### Observation Space
A 6-dimensional vector containing:
- x position
- y position
- orientation (theta)
- x velocity
- y velocity
- angular velocity (omega)

### Action Space
A 2-dimensional vector containing:
- Longitudinal thrust (-MAX_F_LONG to MAX_F_LONG)
- Lateral thrust (-MAX_F_LAT to MAX_F_LAT)

### Reward Function
The reward function provides:
- Small positive reward for each step the agent survives
- Bonus reward based on height achieved
- Negative reward if the drone gets frozen
