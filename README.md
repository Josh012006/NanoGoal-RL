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
  - Touching a wall (any step where the agent's move is blocked by a wall or forced to slide along it) gives a small penalty, so that hugging walls never pays, even when it still makes progress toward the goal
  - Positive reward when the agent reduces the distance between it and the goal
  - Extra bonus (proportional to the gain) each time the agent gets closer to the goal than it has ever been during the episode
  - Small penalty at every step to push the agent to keep moving, larger when it didn't get closer to the goal during that step
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

- [x] Kapturowski, S., Ostrovski, G., Quan, J., Munos, R., & Dabney, W. (2019).
  *Recurrent Experience Replay in Distributed Reinforcement Learning*. ICLR 2019.
  [OpenReview](https://openreview.net/forum?id=r1lyTjAqYX)
  Explains the hidden-state staleness problem in recurrent RL — the LSTM's hidden
  state, captured once during rollout collection, becomes increasingly inconsistent
  with the network's weights as those weights keep updating across training. R2D2
  addresses this with a "burn-in" period that recomputes a fresh hidden state before
  learning from a sequence, rather than reusing the stale one.

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

- **Provider**: DigitalOcean (initial — discontinued after the GitHub Student Developer Pack partnership ended)
- **CPU**: 2 vCPUs
- **RAM**: 4 GiB
- **Storage**: 80 GiB (DigitalOcean droplet)
- **OS**: Ubuntu Server 24.04 LTS

Despite the move to RecurrentPPO (which adds an LSTM to the policy and value networks), the infrastructure above hasn't changed — training still runs entirely on CPU, not GPU. This is deliberate rather than an oversight, for two reasons. First, reliable GPU availability isn't really within reach on a student budget/Azure for Students credits. Second, and more fundamentally, it likely wouldn't help much here even if it were: this project's bottleneck has consistently been CPU-bound environment simulation (Perlin noise topology generation, collision checks, lidar raycasting) and rollout collection across parallel workers, not the size of the neural network doing backprop. Pairing a GPU with fewer than 4 CPUs would mostly leave it idle waiting on single-threaded `SubprocVecEnv` workers to step the environment, rather than meaningfully speeding up training.

## More on the training process

NanoGoal-RL started, on the `v0` branch (https://github.com/Josh012006/NanoGoal-RL/tree/v0), as a first proof of concept: the agent was trained directly on randomly generated and highly varying worlds — easy, medium and hard all mixed together — for only 800,000 timesteps (~1,300 episodes), nowhere near enough to learn so much at once. The agent could reduce its distance to the target and sometimes hold a continuous trajectory, but rarely actually reached it, often spending a whole episode spinning in circles before making any progress. Two things were driving that behavior: the training conditions (the world itself) varied too much from one episode to the next for anything to stabilize, and there was no curriculum — easy, medium and hard were all being learned at the same time instead of progressively.

`v1` addressed both problems. Using the environment's built-in seed-based reproducibility, I hand-picked 20 seeds per difficulty category and introduced an actual curriculum: 100% easy seeds for the easy stage, a 20%/80% easy/medium mix for medium, and a 10%/20%/70% easy/medium/hard mix for hard. I also introduced a growing pool of seeds per stage — starting at 2 and doubling roughly every 2,000 episodes (~1.2M timesteps) — so the set of worlds the agent trained on didn't change too abruptly episode to episode.

`v2` then replaced the hand-picked seeds with an automated, principled classification. `classify_seeds.py` scores 10,000 seeds by running A* on the discrete grid and summing the total angular deviation (turns ≥ 45°) of the optimal path, which captures how many real wall detours a seed requires rather than just its raw distance. Seeds are split into **easy** (< 46° total deviation), **medium** (46°–270°) and **hard** (> 270°) — yielding roughly 5,549 easy, 1,202 medium and 824 hard reachable seeds out of ~7,575 total. The growing-pool idea from v1 was kept but retuned (pools now start at 4, doubling every 700/1,500/3,000 episodes for easy/medium/hard), with an asymmetric 40%/60% training split so easy (larger pool) needs less coverage than medium/hard. The remaining seeds in each category form a held-out test set used exclusively for evaluation in `eval.py`. v2 also brought a wave of infrastructure work: a precomputed topology cache (near-instant resets instead of recomputing Perlin noise every episode), `SubprocVecEnv` parallel environments, a larger rollout buffer, difficulty-scaled `n_epochs`, a fully automated CI/CD training pipeline (GitHub Actions plus a self-hosted systemd runner, with automatic evaluation, plotting and commits), a switch to a deterministic pure-numpy Perlin noise implementation after the third-party `noise` package was found to be non-deterministic across process launches, a critical cache bug fix (available space was being filtered against an empty topology instead of the real one), and trajectory visualization during rendering.

Even with all of this, v2's final model showed a real limitation on hard difficulty: across training, the success rate oscillated between roughly 0.5 and 0.6 with no clear upward trend, and the mean reward actually declined over the course of the run — training for longer didn't help, unlike what was observed for easy and medium. Looking at the agent's behavior, the pattern was consistent: hard seeds often require the agent to make a large turn that momentarily points it away from the target, and because the policy was purely feedforward (PPO with an MLP), it only ever reacts to the CURRENT observation — it has no way to remember that it is mid-detour. A few steps into turning away from a wall, the agent effectively "forgets" why it turned and drifts back toward the same wall it was trying to get around. **That's the problem this version of the project (v3) is trying to solve.**

`v3` attacked that problem by giving the agent a memory. The training algorithm moved from `PPO` to `RecurrentPPO` (`sb3-contrib`, `MultiInputLstmPolicy`), which adds an LSTM to the policy and value networks so that a hidden state persists across the timesteps of an episode, and the lidar was widened (8 → 16 rays, range 20 → 60 grid cells) so walls are detected earlier. Training a recurrent policy stably took more than swapping the class, though: a first easy run saw its success rate peak around 9–10M timesteps and then decline. I attribute it to two things — the LSTM's hidden state being captured once per rollout and then reused, stale, across every PPO epoch (a problem for episodes of up to 800 steps cut into small minibatches), and the policy's entropy collapsing unopposed since `ent_coef` was 0 — and retuned accordingly: `batch_size` 2,000 (larger than an episode), fewer `n_epochs` (8/8/10), `ent_coef=0.01`, `clip_range=0.1` and a linearly decaying learning rate (the final values are in "Training Hyperparameters"). Re-running easy with them fixed the decline. v3 also slowed the seed pools' expansion (1,500/4,000/10,000 episodes for easy/medium/hard), kept a checkpoint every 100,000 timesteps (the last 10), and cut the environment's cost with numba JIT compilation of the lidar, the reset-time navigability check and the wall collisions. On the engineering side, the library code moved into an installable `nanogoal_rl` package.

The results were a clear step forward. On medium, the agent already had a high success rate after only ~29M timesteps (out of a 200M budget) with the seed pool fully covered, so I stopped there. On hard, where `PPO` had plateaued around 0.5–0.6, the recurrent agent now makes large turns around walls, turning its back on the target for a long stretch before being pulled back in: memory (plus the longer lidar) was enough to escape the local reward trap. I stopped the hard run at 151.3M timesteps (out of 400M, ~26 days on the VM) once the success rate stopped rising. The best checkpoint reached about 69 % success on hard seeds, with no more out-of-bounds episodes, and the hard model also secures 95 %+ on easy and medium seeds.

What is left is a different kind of failure. Every remaining failure on hard is a timeout: the agent finds a good route around the walls but stays too close to them and ends up stuck against one a short distance from the target (seed 1296 is a clear example). A quick measurement on 120 hard episodes with the final v3 hard model shows how lopsided it is: successful episodes spend about 5 steps in contact with a wall on average (~1 % of their steps), failed ones about 470 of ~790. And nothing in the reward discourages it — only collisions with blood cells were penalized, never walls. **That's the problem v4 is trying to solve.**

`v4` adds the missing signal: a small penalty (`-0.2`) on every step where the agent's move is blocked by a wall or forces it to slide along one. The value comes from that same measurement: the best progress reward on a free step is about `+0.1`, so hugging a wall never pays; a successful episode loses only about 1 point to it (against `+100` for reaching the goal), while an episode stuck against a wall loses about 90; and an untrained agent, which touches walls 13–17 % of the time, isn't penalized so heavily that early learning gets drowned. Since a new reward changes what every stage learns, the whole curriculum is retrained, starting back from the easy level so that wall avoidance is ingrained in the prior behavior that medium and hard build on. I also made the seed pools grow faster so they reach their full size sooner (v3's hard run alone lasted ~26 days). If the diagnosis is right, the stuck-near-the-target failures should largely disappear and the success rate on hard should move well above v3's ~68 %.

### What changed in v4

- **Added a wall-touch penalty**: the agent now receives `-0.2` on every step where its move overlaps a wall (blocked, or forced to slide along it). The value is the new `__penalty_wall_touch` attribute defined in `NanoEnv.__init__` next to the blood-cell penalties, and it is applied in `step()`; contact is reported by the JIT-compiled collision routine, whose resolved positions are unchanged.
- **Sped up the seed pools' expansion**: easy now expands after 500 episodes (was 1,500), medium now expands after 1,000 episodes (was 4,000) and hard after 2,000 (was 10,000).

## Training Hyperparameters

The table below reflects the `RecurrentPPO` configuration set in `train_easy.py`, `train_medium.py` and `train_hard.py`. Medium and hard each `.load()` the previous stage's checkpoint (`ppo_lstm_easy` → `ppo_lstm_medium` → `ppo_lstm_hard`), so the LSTM-related settings are only actually chosen once, at the easy stage, and simply carried forward through both later stages via the loaded checkpoint.

| Hyperparameter | Easy | Medium | Hard |
|---|---|---|---|
| Policy | `MultiInputLstmPolicy` | `MultiInputLstmPolicy` (loaded from `ppo_lstm_easy`) | `MultiInputLstmPolicy` (loaded from `ppo_lstm_medium`) |
| `n_steps` | `8_000 // n_envs` | `12_000 // n_envs` | `16_000 // n_envs` |
| `batch_size` | 2,000 | 2,000 | 2,000 |
| `n_epochs` | 8 | 8 | 10 |
| `learning_rate` | `LinearSchedule(3e-4 → 5e-5)` | `LinearSchedule(5e-5 → 1e-5)` | `LinearSchedule(1e-5 → 2e-6)` |
| `ent_coef` | 0.01 | 0.01 (inherited) | 0.01 (inherited) |
| `clip_range` | 0.1 | 0.1 (inherited) | 0.1 (inherited) |
| `total_timesteps` | 12,000,000 | 35,000,000 | 400,000,000 |
| `device` | `cpu` | `cpu` | `cpu` |
| `lstm_hidden_size` | 256 (default) | 256 (inherited) | 256 (inherited) |
| `n_lstm_layers` | 1 (default) | 1 (inherited) | 1 (inherited) |
| `shared_lstm` | `False` (default) | `False` (inherited) | `False` (inherited) |
| `enable_critic_lstm` | `True` (default) | `True` (inherited) | `True` (inherited) |
| `gamma` | 0.99 (default) | 0.99 (inherited) | 0.99 (inherited) |
| `gae_lambda` | 0.95 (default) | 0.95 (inherited) | 0.95 (inherited) |
| `vf_coef` | 0.5 (default) | 0.5 (inherited) | 0.5 (inherited) |
| `max_grad_norm` | 0.5 (default) | 0.5 (inherited) | 0.5 (inherited) |

## The results of the training (see `eval.py` for the evaluation code)

Training in progress.
<!-- When all the changes were done, I started training the model. After each training I plotted some interesting relationships between the results parameters.

### Easy mode training
For the easy mode, the model was trained for **~12,000,000 timesteps** (~2.2 days). I preempted the training because all the seeds were covered and the performance was already satisfying. As expected, the training time increased due to the hidden states also being updated. The reassuring part is that the performance of the model is as good it was previously with PPO. Visually, its behavior is also consistent. 

What we can also notice that with the presence of the entropy bonus, the entropy_loss (= -entropy) decreases progressively and finally stabilizes showing that the model has stopped its random exploration by the end of the training.

<table align="center">
  <tr>
    <td align="center">
      <img src="public/easy/success_rate.png" width="800" alt="success rate during learning"><br>
      <u><em>Evolution of success rate during learning episodes</em></u>
    </td>
  </tr>
  <tr><td></td></tr>
  <tr>
    <td align="center">
      <img src="public/easy/pool_increase.png" width="800" alt="pool increase"><br>
      <u><em>The easy level seeds pool's size evolution during training (displayed in %)</em></u>
    </td>
  </tr>
  <tr><td></td></tr>
  <tr>
    <td align="center">
      <img src="public/easy/explained_variance.png" width="800" alt="explained variance during learning"><br>
      <u><em>Evolution of explained variance during learning</em></u>
    </td>
  </tr>
  <tr><td></td></tr>
  <tr>
    <td align="center">
      <img src="public/easy/entropy_loss.png" width="800" alt="entropy loss during learning"><br>
      <u><em>Evolution of entropy</em></u>
    </td>
  </tr>
</table>

<br />

I also evaluate this model on 500 easy level seeds for a more rigorous view on its real deterministic performance :

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


I also tested on medium and hard level seeds to make sure the presence of memory doesn't remove the challenge that those constitute. The challenge remains:

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

Next, I trained the model on medium level seeds, starting from the easy level model as a baseline. Medium worlds require the agent to navigate around 1 to 2 significant obstacles — it must learn when to turn and how to recover its heading after a detour. I originally planned to train it for another **200,000,000 timesteps**, but after **29,000,000 timesteps** (~3.8 days) the agent had already achieved a hight success rate and had seen all the seeds in the pool. So I stopped the training. Here are the training metrics.

<table align="center">
  <tr>
    <td align="center" colspan="2">
      <img src="public/medium/success_rate.png" width="800" alt="success rate during learning"><br>
      <u><em>Evolution of success rate during learning episodes</em></u>
    </td>
  </tr>
  <tr><td></td><td></td></tr>
  <tr>
    <td align="center">
      <img src="public/medium/pool_increase_easy.png" width="800" alt="pool increase"><br>
      <u><em>The easy level seeds pool's size evolution during training (displayed in %)</em></u>
    </td>
    <td align="center">
      <img src="public/medium/pool_increase_medium.png" width="800" alt="pool increase"><br>
      <u><em>The medium level seeds pool's size evolution during training (displayed in %)</em></u>
    </td>
  </tr>
  <tr><td></td><td></td></tr>
  <tr>
    <td align="center" colspan="2">
      <img src="public/medium/explained_variance.png" width="800" alt="explained variance during learning"><br>
      <u><em>Evolution of explained variance during learning</em></u>
    </td>
  </tr>
  <tr><td></td><td></td></tr>
  <tr>
    <td align="center" colspan="2">
      <img src="public/medium/entropy_loss.png" width="800" alt="entropy loss during learning"><br>
      <u><em>Evolution of entropy</em></u>
    </td>
  </tr>
</table>

<br />

And here are the evaluation results :

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


I also tested **Middle schooler Billy :)** on easy and hard tests sets too. We can clearly see more precision on the easy mode and even a somewhat satisfying performance on hard levels. But it still needs some improvements for the hard level. And that's what we are doing next.

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

At last, I launched the hard level training. Like already said in previous versions, the difficulty of hard level seeds is that they require the agent to be able to "give up" on the reward signal momentarily and go around large walls to finally be able to attain its goal. `PPO` wasn't able to do that. And we are now trying to see if `RecurrentPPO` can help. I planned to add **400,000,000 timesteps**. But I had to stop after **151,300,000 timesteps** (lasted ~26 days on my virtual machine). I stopped because even though there was clear learning happening, there wasn't any significant increase on the success rate for hard level seeds at some point. But I discuss below in the "Final analysis" section things that I noticed we can improve to have an even better performance. For now here are the training results : 

<table align="center">
  <tr>
    <td align="center" colspan="3">
      <img src="public/hard/success_rate.png" width="800" alt="success rate during learning"><br>
      <u><em>Evolution of success rate during learning episodes</em></u>
    </td>
  </tr>
  <tr><td></td><td></td><td></td></tr>
  <tr>
    <td align="center">
      <img src="public/hard/pool_increase_easy.png" width="800" alt="pool increase"><br>
      <u><em>The easy level seeds pool's size evolution during training (displayed in %)</em></u>
    </td>
    <td align="center">
      <img src="public/hard/pool_increase_medium.png" width="800" alt="pool increase"><br>
      <u><em>The medium level seeds pool's size evolution during training (displayed in %)</em></u>
    </td>
    <td align="center">
      <img src="public/hard/pool_increase_hard.png" width="800" alt="pool increase"><br>
      <u><em>The hard level seeds pool's size evolution during training (displayed in %)</em></u>
    </td>
  </tr>
  <tr><td></td><td></td><td></td></tr>
  <tr>
    <td align="center" colspan="3">
      <img src="public/hard/explained_variance.png" width="800" alt="explained variance during learning"><br>
      <u><em>Evolution of explained variance during learning</em></u>
    </td>
  </tr>
  <tr><td></td><td></td><td></td></tr>
  <tr>
    <td align="center" colspan="3">
      <img src="public/hard/entropy_loss.png" width="800" alt="entropy loss during learning"><br>
      <u><em>Evolution of entropy</em></u>
    </td>
  </tr>
</table>

<br />

And here are the evaluation results :

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
           alt="Termination to truncation ratio">
      <br>
      <u><em>Termination to truncation ratio per episode</em></u>
    </td>
  </tr>
</table>


We can see a clear imporvement in the performance over hard level seeds. No more out of bounds and mostly timeouts. I also tested **High schooler Billy** on easy and medium level seeds. And not only did he keep his good behaviors, he also improved significantly over the two, securing 95+ % success rate on the two levels :

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

<br />
<br />
 -->

## Final analysis

Coming soon !


## Project structure

```
nanogoal_rl/                         # library code (importable package)
├── __init__.py                      # exposes NanoEnv, registers the "Nano-v0" Gymnasium id
├── env.py                           # NanoEnv: the Gymnasium environment
├── utils.py                         # grid / graph helpers (connectivity, clearance, navigability)
├── perlin_noise.py                  # deterministic Perlin-noise topology generation
├── checkpoint_callback.py           # KeepLastNCheckpoints (SB3 callback)
└── seed_coverage_callback.py        # SeedCoverageCallback (SB3 callback)

train_easy.py, train_medium.py, train_hard.py       # curriculum training
eval.py, visual_eval.py, plots.py, saving_plots.py  # evaluation and plotting
classify_seeds.py, precompute_cache.py, sanity_check_seeds.py  # seed classification and topology cache

seeds.json, assets/                  # data files, read from the working directory
```

The scripts at the root import the library with e.g. `from nanogoal_rl import env` or `from nanogoal_rl.utils import main_related_component`. `import nanogoal_rl` is deliberately light (it does not pull in `torch`/Stable-Baselines3); the two callbacks are imported explicitly from their own modules.

Everything is meant to be run **from the repository root**: `seeds.json`, `topology_cache`, `assets/` and the output folders (`models/`, `checkpoints/`, `logs/`, `results/`, `plots/`, `videos/`) are resolved relative to the working directory.

## Installation

```bash
git clone https://github.com/Josh012006/NanoGoal-RL.git
cd NanoGoal-RL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `nanogoal_rl` package in editable mode (see `pyproject.toml`), so `import nanogoal_rl` works from any folder and code changes are picked up without reinstalling. Exact, reproducible versions stay pinned in `requirements.txt`; `pyproject.toml` only declares lower bounds (`pip install -e ".[train]"` also pulls Stable-Baselines3, SB3-Contrib and Matplotlib). Note that data files (`seeds.json`, `topology_cache`, `assets/`) are still read from the working directory: outside the repository root, `NanoEnv` builds without error but with an empty seed pool, so keep running the scripts from the repository root.

## Usage

All commands below are run from the repository root.

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
- Compare with other RL algorithms
- Multi-agent goal conditioned control

## Author

Josué Mongan

## License

MIT License