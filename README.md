# The agent's behavior over different difficulty levels
<p align="center">
  <img src="public/demo_easy.gif" alt="Demo">
  <br>
  <em>Demonstration of an easy level using a model trained for the easy mode.</em>
</p>

<p align="center">
  <img src="public/demo_medium.gif" alt="Demo">
  <br>
  <em>Demonstration of a medium level using a model trained for the medium mode.</em>
</p>

<p align="center">
  <img src="public/demo_hard.gif" alt="Demo">
  <br>
  <em>Demonstration of a hard level using a model trained for the hard mode.</em>
</p>


# NanoGoal-RL

NanoGoal-RL is a goal-conditioned reinforcement learning project where a simulated 2D nanorobot that I named **Billy** learns to autonomously reach multiple target positions in a continuous environment while avoiding obstacles. The project focuses on decision-making, trajectory optimization, and control using modern reinforcement learning methods.

## Motivation

Controlling robots at very small scales is challenging due to limited sensing, noisy dynamics, and constrained actuation. NanoGoal-RL explores how goal-conditioned reinforcement learning can be used to learn flexible control policies that generalize across many objectives, which is a key requirement for future nano-robotic systems.

## Project Overview

The project simulates a nanorobot moving in a 2D continuous space. At each episode, a target position is randomly generated. The agent receives both its current state and the goal as input and must learn a policy capable of reaching any target efficiently.

Key ideas explored:
- Goal-conditioned reinforcement learning
- Curriculum based learning
- Continuous control
- Autonomous decision-making
- Simulation-based robotics

## Environment

- Observation space: they are mostly normalized to make learning more easy
  - Robot position `(x, y)`
  - Distance to goal `(x_delta_goal, y_delta_goal)`
  - Velocity and orientation relative to the $x$-axis `(v, theta)`
  - Distance to walls in 8 directions from agent
- Action space:
  - Changes to the orientation `dtheta`
  - Variation to the velocity `dv`
- Reward:
  - Negative changes in the velocity and orientation to prevent the agent from spining too much and encourage it to keep a more direct trajectory
  - Touching the white or red cells generated at random places and moving in the blood like liquid gives a penalty (greater penalty for white cells)
  - Positive reward when the agent reduces the distance between it and the goal
  - Positive reward when the goal is reached
  - Negative reward when truncated or the agent goes out of the blood vessel's boundaries (out of the window)
- Episode termination:
  - Goal reached
  - Nanorobot out of the bounds of the environment
  - Maximum number of steps exceeded

## Methods

The agent is trained using Proximal Policy Optimization (PPO).

The implementation relies on standard RL libraries to ensure reproducibility and clarity.

## Technologies Used

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- PyTorch
- TensorBoard
- Tensorflow
- Matplotlib
- Pandas

## More on the training process

In the first version of the project (that you can see on branch `v0` https://github.com/Josh012006/NanoGoal-RL/tree/v0), the model was just trained on randomly generated and highly varying worlds, be it easy, medium or hard mode. Moreaover, it was trained only for 800_000 timesteps (approximatively 1300 complete episodes) which looking back at it, didn't represent much time for learning so much things. 
The consequence wass that even though the model was able to reduce the distance between it and the target and sometimes maintain a continuous trajectory, in most of the cases, it wasn't even reaching the target. The model at that time was taking too many unecessary actions and a lot of time just spent the whole episode spinning in circles before making any progress.

That behavior was caused by two main things : 
- the fact that the environments and other training conditions were changing too much from one episode to another
- the learning wasn't like a curriculum, meaning that the model was learning, on easy, medium and hard mode at the same time

So I did some fine tuning to improve its performance.

The first step was to review the way the environments were chosen in the training process. Fortunately, I had already programmed the environnement so that a specific episode could be entirely reproductible by just passing a seed at the reset time. So I took the time to select : 
- 20 seeds I considered easy mainly because the agent only had to make a straight line to reach the target
- 20 seeds I considered medium. Here the agent only had to learn how to turn around a wall once to see the target more easily
- 20 seeds that are really hard. With those seeds, the agent not only had to cover large distance most of the times but he also had a lot of turns to take around walls in order to achieve the goal
And then I decided of a repartition for the different steps of learning : 
- easy : 100% easy
- medium : 20% easy and 80% medium
- hard : 10% easy, 20% medium and 70% hard

The second step was to fix the way the environments or more precisely the seeds were varying during the training sessions. For that, I used pools of seeds. What I did was I restreined the number of seeds used at different times of the training.
First I started with 2 seeds from the set of seeds for the current difficulty and each 2_000 episodes (approximatively 1_200_000 timesteps), I doubled the size of the pool. What it did was that it made the learning steady and added more stability
to the way the algorithm was infering the policy.


## The results of the training (see `eval.py` for the evaluation code)

When all the changes were done, I started training the model. After each training I plotted some interesting relationships between the results parameters. I want to mention that on the plots, **I truncated some extreme values** that were creating noise 
and preventing me from actually understanding and assessing the model's quality.

### Easy mode training
It lasted **12_000_000 timesteps**. That was **10 hours** in real life. After that stage, **Billy** was able to succeed for almost all the easy worlds of the test set. I was really proud of him. Here were the statistics : 
<p align="center">
  <img src="public/learning_easy.png" width="800" alt="the reward mean during learning"><br>
  <u><em>Evolution of reward during learning episodes</em></u>
</p>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/return-episode-easy.png" width="600"
           alt="Status of reward during testing episodes">
      <br>
      <u><em>Status of reward during testing episodes on easy mode</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/success-episode-easy.png" width="600"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/distances-easy.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/regret-episode.png" width="600"
           alt="final-best">
      <br>
      <u><em>How often it loses progress</em></u>
    </td>
  </tr>
</table>


But I knew he could do more than that. But first before going to the medium and hard modes, I wanted to make sure that there would really be some learning being done.
So I tested **Toddler Billy** on medium and hard tests sets. I only present here the status of the reward during the tests : 

**Test of the model trained for easy mode on medium mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/return-episode-medium.png" width="600"
           alt="Status of reward during test on medium mode seeds">
      <br>
      <u><em>Status of reward during test on medium mode seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/distances-medium.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/easy/success-episode-medium.png" width="450"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />

**Test of the model trained for easy mode on hard mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/return-episode-hard.png" width="600"
           alt="Status of reward during test on hard mode seeds">
      <br>
      <u><em>Status of reward during test on hard mode seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/distances-hard.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/easy/success-episode-hard.png" width="450"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />
<br />

### Medium mode training
I trained the easy model for another **30_000_000 timesteps**. It lasted **18 hours** in real life. Here were the statistics : 
<p align="center">
  <img src="public/learning_medium.png" width="800" alt="the reward mean during learning"><br>
  <u><em>Evolution of reward during learning episodes</em></u>
</p>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/return-episode-medium.png" width="600"
           alt="Status of reward during testing episodes">
      <br>
      <u><em>Status of reward during testing episodes on medium mode</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/success-episode-medium.png" width="600"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/distances-medium.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/regret-episode.png" width="600"
           alt="final-best">
      <br>
      <u><em>How often it loses progress</em></u>
    </td>
  </tr>
</table>


This time I tested **Middle schooler Billy** on easy and hard tests sets too. We can clearly see more precision on the easy mode and even a somewhat satisfying performance on hard levels. But it still needs some improvements for teh hard level. And that's what we are doing next.

**Test of the model trained for medium mode on easy mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/return-episode-easy.png" width="600"
           alt="Status of reward during test on easy mode seeds">
      <br>
      <u><em>Status of reward during test on easy mode seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/distances-easy.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/medium/success-episode-easy.png" width="450"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />

**Test of the model trained for medium mode on hard mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/return-episode-hard.png" width="600"
           alt="Status of reward during test on hard mode seeds">
      <br>
      <u><em>Status of reward during test on hard mode seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/distances-hard.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/medium/success-episode-hard.png" width="450"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />
<br />

### Hard mode training
For the last one I added **78_000_000 timesteps**. That was ** hours** in real life. Here were the statistics : 
<p align="center">
  <img src="public/learning_hard.png" width="800" alt="the reward mean during learning"><br>
  <u><em>Evolution of reward during learning episodes</em></u>
</p>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/return-episode-hard.png" width="600"
           alt="Status of reward during testing episodes">
      <br>
      <u><em>Status of reward during testing episodes on hard mode</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/success-episode-hard.png" width="600"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/distances-hard.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/regret-episode.png" width="600"
           alt="final-best">
      <br>
      <u><em>How often it loses progress</em></u>
    </td>
  </tr>
</table>


Lastly, I tested **High schooler Billy** on easy and medium tests sets too to make sure he didn't forget all he previously learned:

**Test of the model trained for hard mode on easy mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/return-episode-easy.png" width="600"
           alt="Status of reward during test on easy mode seeds">
      <br>
      <u><em>Status of reward during test on easy mode seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/distances-easy.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/hard/success-episode-easy.png" width="450"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>

<br />

**Test of the model trained for hard mode on medium mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/return-episode-medium.png" width="600"
           alt="Status of reward during test on medium mode seeds">
      <br>
      <u><em>Status of reward during test on medium mode seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/distances-medium.png" width="600"
           alt="init-best-final distances">
      <br>
      <u><em>The relationship between initial distance to goal, best distance during episode and final distance at the end</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/hard/success-episode-medium.png" width="450"
           alt="Success rate during episodes">
      <br>
      <u><em>Success rate during episodes</em></u>
    </td>
  </tr>
</table>


## Installation

```bash
git clone https://github.com/your-username/NanoGoal-RL.git
cd NanoGoal-RL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the model for easy mode:
```bash
python train_easy.py
```

Train the model for medium mode:
```bash
python train_medium.py
```

Train the model for hard mode:
```bash
python train_hard.py
```

<br />

Vizualize the learning statistics for easy mode:

```bash
python -m tensorboard.main --logdir logs/<easy_logs_folder>
```

Vizualize the learning statistics for medium mode:

```bash
python -m tensorboard.main --logdir logs/<medium_logs_folder>
```

Vizualize the learning statistics for hard mode:

```bash
python -m tensorboard.main --logdir logs/<hard_logs_folder>
```

<br />

Test a trained model over 100 episodes:
```bash
python eval.py <difficulty_the_model_was_trained_for> <difficulty_of_the_worlds_seeds>
```
where : 
- difficulty_the_model_was_trained_for : 0 for easy, 1 for medium and 2 for hard
- difficulty_of_the_worlds_seeds : 0 for easy seeds, 1 for medium ones, 2 for hard ones and 3 for a mix
The results will appear as CSV files in the results folder.

Vizualize trajectories concerning the performances for the 300 test episodes:
```bash
python plots.py <csv_file_path>
```

<br />

Launch an episode with visual rendering with the trained agent:
```bash
python visual_eval.py <difficulty_the_model_was_trained_for> <difficulty_of_the_world_seed>
```
where : 
- difficulty_the_model_was_trained_for : 0 for easy, 1 for medium and 2 for hard
- difficulty_of_the_worlds_seeds : 0 for the easy seed, 1 for the medium one and 2 for hard one

## Future work

- Add more real-world constraints on the agent. For example represnting the time limit not as a number of steps but as fuel being burned depending on the velocity and orientation variations
- More realistic and complex environments: cell-cell collision management, real CFD(computational fluids dynamics), etc.
- Be more strict on the goal achievement. For example, instead of just trying to attain the target, try to have a low velocity at arrival and a certain orientation
- Extend to 3D control
- Compare with other RL algorithms like HER or DDPG
- Sim-to-real transfer experiments
- Multi-agent goal conditioned control

## Author

Josué Mongan

## License

MIT License
