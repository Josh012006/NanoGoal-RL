# NanoGoalRL

NanoGoalRL is a goal-conditioned reinforcement learning project where a simulated 2D nanorobot learns to autonomously reach multiple target positions in a continuous environment. The project focuses on decision-making, trajectory optimization, and control using modern reinforcement learning methods.

## Motivation

Controlling robots at very small scales is challenging due to limited sensing, noisy dynamics, and constrained actuation. NanoGoalRL explores how goal-conditioned reinforcement learning can be used to learn flexible control policies that generalize across many objectives, which is a key requirement for future nano-robotic systems.

## Project Overview

The project simulates a nanorobot moving in a 2D continuous space. At each episode, a target position is randomly generated. The agent receives both its current state and the goal as input and must learn a policy capable of reaching any target efficiently.

Key ideas explored:
- Goal-conditioned reinforcement learning
- Continuous control
- Autonomous decision-making
- Simulation-based robotics

## Environment

- State space:
  - Robot position `(x, y)`
  - Goal position `(x_goal, y_goal)`
- Action space:
  - Continuous displacement `(dx, dy)`
- Reward:
  - Negative distance to the goal
  - Positive reward when the goal is reached
- Episode termination:
  - Goal reached
  - Maximum number of steps exceeded

## Methods

The agent is trained using actor–critic reinforcement learning algorithms such as:
- Proximal Policy Optimization (PPO)
- (Optional) Soft Actor-Critic (SAC)

The implementation relies on standard RL libraries to ensure reproducibility and clarity.

## Technologies Used

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- PyTorch
- Matplotlib

## Results

The trained agent is able to reach unseen target positions by following smooth and efficient trajectories. Performance is evaluated through trajectory visualization and reward analysis.

## Installation

```bash
git clone https://github.com/your-username/NanoGoalRL.git
cd NanoGoalRL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the agent:
```bash
python train.py
```

Test a trained model:
```bash
python evaluate.py
```

Vizualize trajectories:
```bash
python vizualize.py
```

## Future work

- Add obstacles and complex environments
- Extend to 3D control
- Sim-to-real transfer experiments
- Multi-agent goal conditioned control

## Author

Josué Mongan

## License

MIT License