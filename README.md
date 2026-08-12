# The agent's behavior over different difficulty levels

### Behavior of the model trained for easy mode

<table align="center">
  <tr>
    <td align="center">
      <img src="public/easy/demo_easy_easy.gif" alt="Demo">
      <br>
      <em>Behavior on an easy level world using the model trained for the easy mode.</em>
    </td>
    <td align="center">
      <img src="public/easy/demo_easy_medium.gif" alt="Demo">
      <br>
      <em>Behavior on a medium level world using the model trained for the easy mode.</em>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="public/easy/demo_easy_hard.gif" alt="Demo">
      <br>
      <em>Behavior on a hard level world using the model trained for the easy mode.</em>
    </td>
  </tr>
</table>

<br>

### Behavior of the model trained for medium mode

<table align="center">
  <tr>
    <td align="center">
      <img src="public/medium/demo_medium_easy.gif" alt="Demo">
      <br>
      <em>Behavior on an easy level world using the model trained for the medium mode.</em>
    </td>
    <td align="center">
      <img src="public/medium/demo_medium_medium.gif" alt="Demo">
      <br>
      <em>Behavior on a medium level world using the model trained for the medium mode.</em>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="public/medium/demo_medium_hard.gif" alt="Demo">
      <br>
      <em>Behavior on a hard level world using the model trained for the medium mode.</em>
    </td>
  </tr>
</table>

<br>

### Behavior of the model trained for hard mode

<table align="center">
  <tr>
    <td align="center">
      <img src="public/hard/demo_hard_easy.gif" alt="Demo">
      <br>
      <em>Behavior on an easy level world using the model trained for the hard mode.</em>
    </td>
    <td align="center">
      <img src="public/hard/demo_hard_medium.gif" alt="Demo">
      <br>
      <em>Behavior on a medium level world using the model trained for the hard mode.</em>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="public/hard/demo_hard_hard.gif" alt="Demo">
      <br>
      <em>Behavior on a hard level world using the model trained for the hard mode.</em>
    </td>
  </tr>
</table>


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

## Methods and References

The agent is trained using Proximal Policy Optimization (PPO).

The implementation relies on standard RL libraries to ensure reproducibility and clarity.


Key papers this project builds on — read before implementing the
corresponding phase.

- [x] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms*. arXiv.
  [PDF](https://arxiv.org/pdf/1707.06347.pdf)
  Core policy optimization method — introduces PPO, a stable and sample-efficient
  policy gradient algorithm using a clipped surrogate objective to limit destructive
  policy updates.

- [ ] Hausknecht, M., & Stone, P. (2015).
  *Deep Recurrent Q-Learning for Partially Observable MDPs*. arXiv.
  [PDF](https://arxiv.org/pdf/1507.06527.pdf)
  Core memory architecture — introduces the integration of LSTM networks into deep reinforcement learning to handle partial observability (POMDPs), proving that recurrence allows agents to maintain state history over time when sensors are limited.

## Technologies Used

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- TensorBoard
- Matplotlib
- Pandas
- Pygame

## Training Infrastructure

- **Provider**: DigitalOcean (initial — discontinued after the GitHub Student Developer Pack partnership ended), Microsoft Azure (current, via Azure for Students)
- **CPU**: 2 vCPUs
- **RAM**: 4 GiB
- **Storage**: 80 GiB (DigitalOcean droplet) / separate OS disk + 128 GiB data disk (Azure)
- **OS**: Ubuntu Server 24.04 LTS

## More on the training process

In the first version of the project (that you can see on branch `v0` https://github.com/Josh012006/NanoGoal-RL/tree/v0), the model was just trained on randomly generated and highly varying worlds, be it easy, medium or hard mode. Moreover, it was trained only for 800_000 timesteps (approximatively 1300 complete episodes) which looking back at it, didn't represent much time for learning so much things. 
The consequence was that even though the model was able to reduce the distance between it and the target and sometimes maintain a continuous trajectory, in most of the cases, it wasn't even reaching the target. The model at that time was taking too many unecessary actions and a lot of time just spent the whole episode spinning in circles before making any progress.

That behavior was caused by two main things : 
- the fact that the environments and other training conditions were changing too much from one episode to another
- the learning wasn't like a curriculum, meaning that the model was learning, on easy, medium and hard mode at the same time

So I did some fine tuning to improve its performance.

The first step was to review the way the environments were chosen in the training process. Fortunately, I had already programmed the environnement so that a specific episode could be entirely reproductible by just passing a seed at the reset time. Instead of manually picking seeds, I wrote a script (`classify_seeds.py`) that automatically scores 10,000 seeds using the A* algorithm on the discrete grid. For each seed, A* computes the total angular deviation of the optimal path — that is, the sum of all significant direction changes (≥ 45°) along the path from agent to target. This captures not just the distance but also how many times the agent must navigate around walls. Seeds where no path exists are discarded. The remaining reachable seeds are partitioned into three difficulty categories based on this deviation:

- **Easy** (< 46° total deviation): the agent can reach the target with near-straight-line navigation and no real wall detours
- **Medium** (46°–270°): the agent must learn to navigate around 1 to 2 significant obstacles
- **Hard** (> 270°): the agent must combine multiple navigation skills to handle complex, multi-detour paths

And then I decided of a repartition for the different steps of learning :
- easy : 100% easy
- medium : 20% easy and 80% medium
- hard : 10% easy, 20% medium and 70% hard

The second step was to fix the way the environments or more precisely the seeds were varying during the training sessions. For that, I used pools of seeds. What I did was I restreined the number of seeds used at different times of the training.
I started with a pool of 4 seeds from the set of seeds for the current difficulty and doubled the size of the pool at regular intervals — every 700 episodes for easy, 1500 for medium, and 3000 for hard. This made learning steady and added more stability to the way the algorithm was inferring the policy.

For training, 40% of easy seeds are sampled randomly, while 60% of medium and hard seeds are used — the latter two categories have smaller pools so a larger fraction is needed to give the agent sufficient variety. The remaining seeds in each category form a held-out test set used exclusively for evaluation in `eval.py`, ensuring that the reported performance metrics reflect genuine generalization and not memorization of training environments.

### What changed in v2

Several infrastructure and training improvements were made for this version:

- **Improved seed classification with angular deviation**: difficulty is now measured by the total angular deviation of the A* path (sum of direction changes ≥ 45°), not just path length. This ensures easy seeds require no wall detours, medium seeds require 1–2, and hard seeds require complex multi-detour navigation. Out of ~7,575 reachable seeds, this yields ~5,549 easy, ~1,202 medium, and ~824 hard seeds.
- **Asymmetric training split**: easy uses 40% of its seeds for training (large pool, less variety needed), while medium and hard use 60% to compensate for their smaller pools.
- **Precomputed topology cache**: vessel topologies and free spaces are now precomputed once and stored on disk. This makes episode resets nearly instant instead of recomputing expensive Perlin noise maps at each episode, significantly reducing overhead.
- **Parallel environments**: training now uses `SubprocVecEnv` to run 2 environments in parallel (one per CPU), doubling the data collection throughput. The number of environments is detected automatically from the available CPU count.
- **Larger rollout buffer**: `n_steps` was doubled to 20,000 per environment to reduce the proportion of time spent in backpropagation relative to rollout collection, keeping both CPUs more consistently busy.
- **Increased n_epochs by difficulty**: easy uses `n_epochs=10` (default), medium uses `n_epochs=15`, and hard uses `n_epochs=20`. More passes per rollout allow the model to extract more signal from complex episodes, acting as an implicit replay ratio increase for harder stages.
- **Automated CI/CD training pipeline**: training is now triggered via GitHub Actions and runs as a systemd service on a self-hosted Digital Ocean droplet, completely decoupled from the runner lifecycle. The pipeline handles dependency installation, cache management, training, evaluation, plot generation, and commits results automatically. Email notifications are sent at key training milestones via SendGrid.
- **Deterministic topology generation**: the third-party `noise` package was found to produce non-deterministic output across separate process launches for certain (base, octaves) combinations, silently corrupting seed-based difficulty classification and reproducibility. It was replaced with `perlin_noise.py`, a fully deterministic, vectorized, pure-numpy implementation — also ~25× faster.
- **Critical bug fix in the topology cache builder**: `precompute_cache.py`'s clearance filter was checking wall proximity against an empty (all-zero) topology instead of the seed's actual generated topology, because the generated topology was never assigned to the environment instance before filtering. This silently gave every cached seed a wrong `available_space` (different count and content than a correct fresh computation), which cascaded into different agent/target/cell placements whenever a seed was served from the cache — making training and evaluation results irreproducible in a way that was very difficult to trace back to its source. Fixed by assigning the topology to the environment before filtering, matching `env.py`'s own reset() logic exactly.
- **Trajectory visualization**: the agent's full path is now drawn as a dashed line during rendering, making it much easier to visually assess how it navigates toward the goal (or gets stuck) over the course of an episode.

## The results of the training (see `eval.py` for the evaluation code)

When all the changes were done, I started training the model. After each training I plotted some interesting relationships between the results parameters.

### Easy mode training
It lasted **12,000,000 timesteps** (~10 hours) — training was stopped early once TensorBoard/WandB metrics showed the model had already converged, well short of the original 50,000,000 timestep budget. Easy worlds require no wall detours — the agent only needs to learn to navigate in a near-straight line toward the target. After this stage, **Billy** was able to succeed for more than half of the easy worlds of the test set.

<p align="center">
  <img src="public/easy/reward_mean.png" width="800" alt="the reward mean during learning"><br>
  <u><em>Evolution of reward during learning episodes</em></u>
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="public/easy/success_rate.png" width="800" alt="success rate during learning"><br>
      <u><em>Evolution of success rate during learning episodes</em></u>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="public/easy/explained_variance.png" width="800" alt="explained variance during learning"><br>
      <u><em>Evolution of explained variance during learning — stays consistently above 0.92, indicating the value function learned well</em></u>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="public/easy/entropy_loss.png" width="800" alt="entropy loss during learning"><br>
      <u><em>Evolution of entropy — rises as the pool of seeds expands and the agent explores more diverse strategies</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/return-episode-easy.png" width="600"
           alt="Return distribution on easy test seeds">
      <br>
      <u><em>Return distribution on easy test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/success-episode-easy.png" width="600"
           alt="Success rate on easy test seeds">
      <br>
      <u><em>Success rate on easy test seeds</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/distances-easy.png" width="600"
           alt="Initial distance vs best reached distance">
      <br>
      <u><em>Initial distance vs best reached distance per episode</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/regret-episode.png" width="600"
           alt="Regret distribution">
      <br>
      <u><em>Regret distribution — how much progress is lost by the end of each episode</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/terminated-truncated.png" width="600"
           alt="Termination to truncation ratio">
      <br>
      <u><em>Termination to truncation ratio per episode</em></u>
    </td>
  </tr>
</table>


But I knew he could do more than that. But first before going to the medium and hard modes, I wanted to make sure that there would really be some learning being done.
So I tested **Toddler Billy** on medium and hard tests sets. I only present here the return and distance plots :

**Test of the model trained for easy mode on medium mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/return-episode-medium.png" width="600"
           alt="Return distribution on medium test seeds">
      <br>
      <u><em>Return distribution on medium test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/distances-medium.png" width="600"
           alt="Initial distance vs best reached distance on medium seeds">
      <br>
      <u><em>Initial distance vs best reached distance on medium seeds</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/easy/success-episode-medium.png" width="450"
           alt="Success rate on medium test seeds">
      <br>
      <u><em>Success rate on medium test seeds</em></u>
    </td>
  </tr>
</table>

<br />

**Test of the model trained for easy mode on hard mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/easy/return-episode-hard.png" width="600"
           alt="Return distribution on hard test seeds">
      <br>
      <u><em>Return distribution on hard test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/easy/distances-hard.png" width="600"
           alt="Initial distance vs best reached distance on hard seeds">
      <br>
      <u><em>Initial distance vs best reached distance on hard seeds</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/easy/success-episode-hard.png" width="450"
           alt="Success rate on hard test seeds">
      <br>
      <u><em>Success rate on hard test seeds</em></u>
    </td>
  </tr>
</table>

<br />
<br />

### Medium mode training
I trained the easy model for another **150,000,000 timesteps** (~5 days) but this time on medium level seeds. Medium worlds require the agent to navigate around 1 to 2 significant obstacles — it must learn when to turn and how to recover its heading after a detour.

<p align="center">
  <img src="public/medium/reward_mean.png" width="800" alt="the reward mean during learning"><br>
  <u><em>Evolution of reward during learning episodes</em></u>
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="public/medium/success_rate.png" width="800" alt="success rate during learning"><br>
      <u><em>Evolution of success rate during learning episodes</em></u>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="public/medium/explained_variance.png" width="800" alt="explained variance during learning"><br>
      <u><em>Evolution of explained variance during learning — stays consistently above 0.92, indicating the value function learned well</em></u>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="public/medium/entropy_loss.png" width="800" alt="entropy loss during learning"><br>
      <u><em>Evolution of entropy — rises as the pool of seeds expands and the agent explores more diverse strategies</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/return-episode-medium.png" width="600"
           alt="Return distribution on medium test seeds">
      <br>
      <u><em>Return distribution on medium test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/success-episode-medium.png" width="600"
           alt="Success rate on medium test seeds">
      <br>
      <u><em>Success rate on medium test seeds</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/distances-medium.png" width="600"
           alt="Initial distance vs best reached distance on medium seeds">
      <br>
      <u><em>Initial distance vs best reached distance on medium seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/regret-episode.png" width="600"
           alt="Regret distribution">
      <br>
      <u><em>Regret distribution</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/terminated-truncated.png" width="600"
           alt="Termination to truncation ratio">
      <br>
      <u><em>Termination to truncation ratio per episode</em></u>
    </td>
  </tr>
</table>


This time I tested **Middle schooler Billy** on easy and hard tests sets too. We can clearly see more precision on the easy mode and even a somewhat satisfying performance on hard levels. But it still needs some improvements for the hard level. And that's what we are doing next.

**Test of the model trained for medium mode on easy mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/return-episode-easy.png" width="600"
           alt="Return distribution on easy test seeds">
      <br>
      <u><em>Return distribution on easy test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/distances-easy.png" width="600"
           alt="Initial distance vs best reached distance on easy seeds">
      <br>
      <u><em>Initial distance vs best reached distance on easy seeds</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/medium/success-episode-easy.png" width="450"
           alt="Success rate on easy test seeds">
      <br>
      <u><em>Success rate on easy test seeds</em></u>
    </td>
  </tr>
</table>

<br />

**Test of the model trained for medium mode on hard mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/medium/return-episode-hard.png" width="600"
           alt="Return distribution on hard test seeds">
      <br>
      <u><em>Return distribution on hard test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/medium/distances-hard.png" width="600"
           alt="Initial distance vs best reached distance on hard seeds">
      <br>
      <u><em>Initial distance vs best reached distance on hard seeds</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/medium/success-episode-hard.png" width="450"
           alt="Success rate on hard test seeds">
      <br>
      <u><em>Success rate on hard test seeds</em></u>
    </td>
  </tr>
</table>

<br />
<br />

### Hard mode training
For the last step, I added **400,000,000 timesteps** (~12 days). Hard worlds require the agent to combine everything it has learned — navigating around multiple significant obstacles (> 270° total angular deviation) while maintaining directional progress toward a distant goal.

<p align="center">
  <img src="public/hard/reward_mean.png" width="800" alt="the reward mean during learning"><br>
  <u><em>Evolution of reward during learning episodes</em></u>
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="public/hard/success_rate.png" width="800" alt="success rate during learning"><br>
      <u><em>Evolution of success rate during learning episodes</em></u>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="public/hard/explained_variance.png" width="800" alt="explained variance during learning"><br>
      <u><em>Evolution of explained variance during learning — stays consistently above 0.92, indicating the value function learned well</em></u>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="public/hard/entropy_loss.png" width="800" alt="entropy loss during learning"><br>
      <u><em>Evolution of entropy — rises as the pool of seeds expands and the agent explores more diverse strategies</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/return-episode-hard.png" width="600"
           alt="Return distribution on hard test seeds">
      <br>
      <u><em>Return distribution on hard test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/success-episode-hard.png" width="600"
           alt="Success rate on hard test seeds">
      <br>
      <u><em>Success rate on hard test seeds</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/distances-hard.png" width="600"
           alt="Initial distance vs best reached distance on hard seeds">
      <br>
      <u><em>Initial distance vs best reached distance on hard seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/regret-episode.png" width="600"
           alt="Regret distribution">
      <br>
      <u><em>Regret distribution</em></u>
    </td>
  </tr>
</table>

<br />

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/terminated-truncated.png" width="600"
           alt="Termination and truncation ratio">
      <br>
      <u><em>Termination to truncation ratio per episode</em></u>
    </td>
  </tr>
</table>


Lastly, I tested **High schooler Billy** on easy and medium tests sets too to make sure he didn't forget all he previously learned:

**Test of the model trained for hard mode on easy mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/return-episode-easy.png" width="600"
           alt="Return distribution on easy test seeds">
      <br>
      <u><em>Return distribution on easy test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/distances-easy.png" width="600"
           alt="Initial distance vs best reached distance on easy seeds">
      <br>
      <u><em>Initial distance vs best reached distance on easy seeds</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/hard/success-episode-easy.png" width="450"
           alt="Success rate on easy test seeds">
      <br>
      <u><em>Success rate on easy test seeds</em></u>
    </td>
  </tr>
</table>

<br />

**Test of the model trained for hard mode on medium mode worlds**

<table align="center">
  <tr>
    <td align="center">
      <img src="plots/hard/return-episode-medium.png" width="600"
           alt="Return distribution on medium test seeds">
      <br>
      <u><em>Return distribution on medium test seeds</em></u>
    </td>
    <td align="center">
      <img src="plots/hard/distances-medium.png" width="600"
           alt="Initial distance vs best reached distance on medium seeds">
      <br>
      <u><em>Initial distance vs best reached distance on medium seeds</em></u>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/hard/success-episode-medium.png" width="450"
           alt="Success rate on medium test seeds">
      <br>
      <u><em>Success rate on medium test seeds</em></u>
    </td>
  </tr>
</table>

## Final analysis

The training metrics of the final model obtained show that **PPO alone struggles to recover an optimal behavior on hard difficulty environments**. Throughout the training, the success rate oscillates between 0.5 and 0.6 and the reward mean decreases. There isn't a clear improvement on the behavior of agent even after 400M steps, opposite to what was observed for the two previous training phases (easy and medium difficulty). The evaluation process and the [visual behavior](#behavior-of-the-model-trained-for-hard-mode) of the agent both confirm that the agent didn't learn any new useful behavior but worse, it lost part of its useful pre-learned conduct.

It's explainable when we look at the challenge the agent is faced with for the hard level difficulty. The target is most of the time situated in a place that requires the agent to make a big turn involving turning its back on it. The observation only gives info on the distance to the target at a given time and the lidar doesn't always cover enough distance for the agent to know in advance that their is wall separating it from the target in the direction it has taken. With PPO, that optimizes the reward given those observation, the agent always comes back to the same wall because even if it tries to turn back, after some timesteps it already forgets that it is in a phase of escaping. 

I argue that this limitation should be surmountable by using `RecurrentPPO`, a version of `PPO` that integrates `LSTM` to have a memory of the past states the agent has been in. Increasing the radius the pseudo-lidar covers should also help the agent detect the walls earlier and take meaningful actions to go around them. These changes, alongside some other improvements will be studied in a new version of the project on the [v3](https://github.com/Josh012006/NanoGoal-RL/tree/v3) branch.


## Installation

```bash
git clone https://github.com/Josh012006/NanoGoal-RL.git
cd NanoGoal-RL
python3 -m venv venv
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
tensorboard --logdir logs/<easy_logs_folder>
```

Vizualize the learning statistics for medium mode:

```bash
tensorboard --logdir logs/<medium_logs_folder>
```

Vizualize the learning statistics for hard mode:

```bash
tensorboard --logdir logs/<hard_logs_folder>
```

<br />

Test a trained model over 100 episodes:
```bash
python eval.py --model {easy,medium,hard} --seed {easy,medium,hard,mix}
```
where : 
- `--model` : difficulty the model was trained for
- `--seed` : difficulty of the world seeds to test on (`mix` combines all three categories)
The results will appear as CSV files in the results folder.

Vizualize trajectories concerning the performances for the 100 test episodes:
```bash
python plots.py <csv_file_path>
```

<br />

Launch an episode with visual rendering with the trained agent:
```bash
python visual_eval.py --model {easy,medium,hard} --seed {easy,medium,hard}
```
where : 
- `--model` : difficulty the model was trained for
- `--seed` : difficulty of the world seed to use for the episode

To inspect one exact seed instead (e.g. a seed pulled from a results CSV):
```bash
python visual_eval.py --model easy --seed_value 3271
```

Every episode is also saved as an animated GIF in `videos/`, whether or not a real display is attached — on a headless server, set `SDL_VIDEODRIVER=dummy` first so pygame doesn't try to open a real window:
```bash
SDL_VIDEODRIVER=dummy python visual_eval.py --model easy --seed_value 3271
```

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