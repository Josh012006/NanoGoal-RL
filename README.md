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
- Partially observable Markov decision processes (POMDPs) — the agent never sees the full world, only a local lidar reading and its position relative to the goal
- Memory-augmented policies — giving the agent a recurrent hidden state so it can remember what it was doing (e.g. mid-detour around a wall) instead of reacting only to the current observation

## Environment

- Observation space: they are mostly normalized to make learning more easy
  - Robot position `(x, y)`
  - Distance to goal `(x_delta_goal, y_delta_goal)`
  - Velocity and orientation relative to the $x$-axis `(v, theta)`
  - Distance to walls in 16 directions from agent
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

As of v3, the agent is trained using Recurrent Proximal Policy Optimization (RecurrentPPO), via the `sb3-contrib` library. This is standard PPO with an LSTM (Long Short-Term Memory) layer added inside the policy and value networks, giving the agent a hidden state that persists across timesteps within an episode. Instead of reacting only to the current observation, the agent can now carry information forward — such as "I am currently escaping a wall" — which earlier versions of the project (plain PPO, purely feedforward) had no way to represent.

The implementation relies on standard RL libraries to ensure reproducibility and clarity.


Key papers this project builds on — read before implementing the
corresponding phase.

- [x] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms*. arXiv.
  [PDF](https://arxiv.org/pdf/1707.06347.pdf)
  Core policy optimization method — introduces PPO, a stable and sample-efficient
  policy gradient algorithm using a clipped surrogate objective to limit destructive
  policy updates.

- [x] Hausknecht, M., & Stone, P. (2015).
  *Deep Recurrent Q-Learning for Partially Observable MDPs*. arXiv.
  [PDF](https://arxiv.org/pdf/1507.06527.pdf)
  Core memory architecture — introduces the integration of LSTM networks into deep reinforcement learning to handle partial observability (POMDPs), proving that recurrence allows agents to maintain state history over time when sensors are limited.

## Technologies Used

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- SB3-Contrib
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

Despite the move to RecurrentPPO (which adds an LSTM to the policy and value networks), the infrastructure above hasn't changed — training still runs entirely on CPU, not GPU. This is deliberate rather than an oversight, for two reasons. First, reliable GPU availability isn't really within reach on a student budget/Azure for Students credits. Second, and more fundamentally, it likely wouldn't help much here even if it were: this project's bottleneck has consistently been CPU-bound environment simulation (Perlin noise topology generation, collision checks, lidar raycasting) and rollout collection across parallel workers, not the size of the neural network doing backprop. Pairing a GPU with fewer than 4 CPUs would mostly leave it idle waiting on single-threaded `SubprocVecEnv` workers to step the environment, rather than meaningfully speeding up training.

## More on the training process

NanoGoal-RL started, on the `v0` branch (https://github.com/Josh012006/NanoGoal-RL/tree/v0), as a first proof of concept: the agent was trained directly on randomly generated and highly varying worlds — easy, medium and hard all mixed together — for only 800,000 timesteps (~1,300 episodes), nowhere near enough to learn so much at once. The agent could reduce its distance to the target and sometimes hold a continuous trajectory, but rarely actually reached it, often spending a whole episode spinning in circles before making any progress. Two things were driving that behavior: the training conditions (the world itself) varied too much from one episode to the next for anything to stabilize, and there was no curriculum — easy, medium and hard were all being learned at the same time instead of progressively.

`v1` addressed both problems. Using the environment's built-in seed-based reproducibility, I hand-picked 20 seeds per difficulty category and introduced an actual curriculum: 100% easy seeds for the easy stage, a 20%/80% easy/medium mix for medium, and a 10%/20%/70% easy/medium/hard mix for hard. I also introduced a growing pool of seeds per stage — starting at 2 and doubling roughly every 2,000 episodes (~1.2M timesteps) — so the set of worlds the agent trained on didn't change too abruptly episode to episode.

`v2` then replaced the hand-picked seeds with an automated, principled classification. `classify_seeds.py` scores 10,000 seeds by running A* on the discrete grid and summing the total angular deviation (turns ≥ 45°) of the optimal path, which captures how many real wall detours a seed requires rather than just its raw distance. Seeds are split into **easy** (< 46° total deviation), **medium** (46°–270°) and **hard** (> 270°) — yielding roughly 5,549 easy, 1,202 medium and 824 hard reachable seeds out of ~7,575 total. The growing-pool idea from v1 was kept but retuned (pools now start at 4, doubling every 700/1,500/3,000 episodes for easy/medium/hard), with an asymmetric 40%/60% training split so easy (larger pool) needs less coverage than medium/hard. The remaining seeds in each category form a held-out test set used exclusively for evaluation in `eval.py`. v2 also brought a wave of infrastructure work: a precomputed topology cache (near-instant resets instead of recomputing Perlin noise every episode), `SubprocVecEnv` parallel environments, a larger rollout buffer, difficulty-scaled `n_epochs`, a fully automated CI/CD training pipeline (GitHub Actions plus a self-hosted systemd runner, with automatic evaluation, plotting and commits), a switch to a deterministic pure-numpy Perlin noise implementation after the third-party `noise` package was found to be non-deterministic across process launches, a critical cache bug fix (available space was being filtered against an empty topology instead of the real one), and trajectory visualization during rendering.

Even with all of this, v2's final model showed a real limitation on hard difficulty: across training, the success rate oscillated between roughly 0.5 and 0.6 with no clear upward trend, and the mean reward actually declined over the course of the run — training for longer didn't help, unlike what was observed for easy and medium. Looking at the agent's behavior, the pattern was consistent: hard seeds often require the agent to make a large turn that momentarily points it away from the target, and because the policy was purely feedforward (PPO with an MLP), it only ever reacts to the CURRENT observation — it has no way to remember that it is mid-detour. A few steps into turning away from a wall, the agent effectively "forgets" why it turned and drifts back toward the same wall it was trying to get around. **That's the problem this version of the project (v3) is trying to solve.**

### What changed in v3

- **Switched the training algorithm from PPO to RecurrentPPO**: `"MultiInputPolicy"` → `"MultiInputLstmPolicy"` (via `sb3-contrib`), adding an LSTM to the policy and value networks so the agent can carry a hidden state across timesteps within an episode. This is the direct attempt at fixing the memorylessness problem described above (see Hausknecht & Stone, 2015, DRQN, in Methods and References).
- **Widened the agent's perception**: the lidar now casts 16 rays instead of 8, with its range extended from 20 to 60 grid cells, so walls can be detected earlier and from more directions.
- **Slowed the curriculum's seed-pool expansion** (700/1,500/3,000 → 1,500/4,000/10,000 episodes for easy/medium/hard) to keep each pool size stable for longer given the added recurrent state.
- **Increased the medium-difficulty training budget** from 150M to 200M timesteps.
- **Training still runs entirely on CPU** (see Training Infrastructure above) — the switch to RecurrentPPO didn't come with a move to GPU.
- **Renamed saved model files** from `ppo_nanogoal_*` to `ppo_lstm_*` to make the architecture explicit and avoid ever loading a v2-era (non-recurrent) checkpoint into RecurrentPPO by mistake — the two architectures are not compatible, so `easy` had to be retrained from scratch under v3 before `medium`/`hard` can chain off it again.
- **`eval.py`/`visual_eval.py` now manage the LSTM's hidden state explicitly**: reset at the start of every episode, carried across steps within it — required by `RecurrentPPO`, meaningless for plain `PPO`.
- **Refactored `saving_plots.py`'s CLI**: replaced positional numeric arguments (`0`/`1`/`2` for difficulty, a raw `0`/`1` flag) with named `--model`/`--seed` options mirroring `eval.py`/`visual_eval.py`'s own convention, removing an entire class of hard-to-read, easy-to-mis-order invocations.
- **Reworked the seed-coverage metrics**: `SeedCoverageCallback` no longer logs the raw `unique_seen` seed count (kept only the normalized `pct_unique_seen`), and now also logs a live per-category `success_rate` — the fraction of episodes on already-seen easy/medium/hard seeds that ended in success — giving more direct visibility into curriculum progress than seed coverage alone.
- **Pinned `numpy` to `2.4.1` instead of `2.4.0`**: `2.4.0` was yanked from PyPI shortly after release over a backward-compatibility bug (a typo in `SeedlessSequence` breaking wheels built against `numpy < 2.4.0` via the `random` Cython API), which pip surfaces as an install-time warning. `2.4.1` is the immediate patch release that fixes exactly that bug and nothing else.

## Training Hyperparameters

The table below reflects the `RecurrentPPO` configuration set in `train_easy.py`, `train_medium.py` and `train_hard.py`. Medium and hard each `.load()` the previous stage's checkpoint (`ppo_lstm_easy` → `ppo_lstm_medium` → `ppo_lstm_hard`), so the LSTM-related settings are only actually chosen once, at the easy stage, and simply carried forward through both later stages via the loaded checkpoint.

| Hyperparameter | Easy | Medium | Hard |
|---|---|---|---|
| Policy | `MultiInputLstmPolicy` | `MultiInputLstmPolicy` (loaded from `ppo_lstm_easy`) | `MultiInputLstmPolicy` (loaded from `ppo_lstm_medium`) |
| `n_steps` | `20_000 // n_envs` | `20_000 // n_envs` | `20_000 // n_envs` |
| `batch_size` | 200 | 200 (inherited, not overridden) | 200 (inherited, not overridden) |
| `n_epochs` | 10 | 15 | 20 |
| `learning_rate` | 3e-4 (default) | 1e-4 | 5e-5 |
| `total_timesteps` | 20,000,000 | 200,000,000 | 400,000,000 |
| `device` | `cpu` | `cpu` | `cpu` |
| `lstm_hidden_size` | 256 (default) | 256 (inherited) | 256 (inherited) |
| `n_lstm_layers` | 1 (default) | 1 (inherited) | 1 (inherited) |
| `shared_lstm` | `False` (default) | `False` (inherited) | `False` (inherited) |
| `enable_critic_lstm` | `True` (default) | `True` (inherited) | `True` (inherited) |
| `gamma` | 0.99 (default) | 0.99 (inherited) | 0.99 (inherited) |
| `gae_lambda` | 0.95 (default) | 0.95 (inherited) | 0.95 (inherited) |
| `clip_range` | 0.2 (default) | 0.2 (inherited) | 0.2 (inherited) |
| `ent_coef` | 0.0 (default) | 0.0 (inherited) | 0.0 (inherited) |
| `vf_coef` | 0.5 (default) | 0.5 (inherited) | 0.5 (inherited) |
| `max_grad_norm` | 0.5 (default) | 0.5 (inherited) | 0.5 (inherited) |

## The results of the training (see `eval.py` for the evaluation code)

When all the changes were done, I started training the model. After each training I plotted some interesting relationships between the results parameters.

### Easy mode training
It lasted **20,000,000 timesteps** (~10 hours) — training was stopped early once TensorBoard/WandB metrics showed the model had already converged, well short of the original 50,000,000 timestep budget. Easy worlds require no wall detours — the agent only needs to learn to navigate in a near-straight line toward the target. After this stage, **Billy** was able to succeed for more than half of the easy worlds of the test set.

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
I trained the easy model for another **200,000,000 timesteps** (~5 days) but this time on medium level seeds. Medium worlds require the agent to navigate around 1 to 2 significant obstacles — it must learn when to turn and how to recover its heading after a detour.

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

Coming soon.


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

Save the same plots to disk instead of popping up an interactive window (used automatically by the CI training pipeline after each `eval.py` run):
```bash
python saving_plots.py --model {easy,medium,hard} --seed {easy,medium,hard,mix}
```
where :
- `--model` : difficulty the model was trained for
- `--seed` : difficulty of the world seeds the model was evaluated on (`mix` combines all three)

By default this reads `results/<model>/ppo_eval_<seed>.csv` (matching `eval.py`'s own output path) and saves plots to `plots/<model>/`. Both can be overridden with `--csv <path>` and `--output-folder <path>` if needed. When `--model` and `--seed` match (the model's "native" evaluation), a few extra diagnostic plots are also generated (termination breakdown, episode length, regret) in addition to the return/success/distance plots shared by every evaluation.

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