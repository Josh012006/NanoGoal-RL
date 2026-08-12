import os
import torch

# Force single-threaded, bit-reproducible PyTorch computation. Multi-threaded
# BLAS/OpenMP reductions (matrix multiplies inside the policy network) are NOT
# guaranteed bit-identical run to run, even on the same machine, because the
# order partial sums from different threads get combined depends on OS thread
# scheduling. In this environment, tiny floating-point differences can flip a
# discrete decision (e.g. which side of a wall-collision branch is taken),
# which then compounds over the episode into a completely different rollout.
# This must be set before any PPO model is created or loaded.
torch.set_num_threads(1)

import env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib import RecurrentPPO
from checkpoint_callback import KeepLastTwoCheckpoints
from seed_coverage_callback import SeedCoverageCallback


def make_env(worker_idx):
    # worker_idx gives each parallel SubprocVecEnv worker a distinct
    # per-episode sampling stream (see env.py's worker_seed_offset), so
    # parallel workers explore genuinely different seeds instead of following
    # the same draw sequence in lockstep. Monitor is added manually here
    # (make_vec_env normally does this for us) since we build SubprocVecEnv
    # directly to be able to pass a different worker_idx to each instance.
    def _init():
        return Monitor(env.NanoEnv(difficulty="easy", worker_seed_offset=worker_idx))
    return _init


if __name__ == "__main__":
    # Check the environment first
    check_env(env.NanoEnv(difficulty="easy"))

    # Automatically detect the number of available CPUs
    n_envs = min(os.cpu_count(), 8)
    n_steps = 20_000 // n_envs  # increased to reduce backprop proportion vs rollout
    print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

    # SubprocVecEnv spawns one process per env, enabling true CPU parallelism
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Use RUN_ID from environment for unique checkpoint folder per run
    run_id = os.environ.get("RUN_ID", "local")
    checkpoint_path = f"./checkpoints/easy/{run_id}/"
    print(f"Checkpoint path: {checkpoint_path}")

    checkpoint_callback = KeepLastTwoCheckpoints(
        save_freq=1_000_000,
        save_path=checkpoint_path,
        name_prefix="ppo_easy"
    )

    # Tracks real per-seed training coverage (how many individual seeds from
    # each pool have actually been visited, and how often) — logged every
    # 100,000 timesteps to TensorBoard/WandB, with a final histogram per pool
    # saved to plots/easy/ once training ends.
    seed_coverage_callback = SeedCoverageCallback(
        log_freq=100_000,
        output_dir="plots/easy"
    )

    # ── Optional Weights & Biases logging ─────────────────────────────────────
    # Lets training be monitored remotely from wandb.ai while it runs on the
    # droplet, in addition to TensorBoard. sync_tensorboard=True reuses the
    # exact same metrics already written to tensorboard_log below, so no
    # separate logging code is needed. If wandb isn't configured (no
    # WANDB_API_KEY, no network), training still proceeds normally with
    # TensorBoard alone — this must never be able to crash a training run.
    wandb_run = None
    wandb_callback = None
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(
            project="nanogoal-rl",
            name=f"easy_{run_id}",
            group=run_id,
            job_type="easy",
            dir="./logs",
            sync_tensorboard=True,
        )
        wandb_callback = WandbCallback(verbose=2)
        print("wandb logging enabled.")
    except Exception as e:
        print(f"wandb unavailable, continuing with TensorBoard only: {e}")

    callbacks = [checkpoint_callback, seed_coverage_callback]
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    # Define and train the agent
    # v3: switched from PPO/MultiInputPolicy to RecurrentPPO/MultiInputLstmPolicy
    # (sb3-contrib). The README's "Final analysis" of the v2 hard-mode results
    # showed the agent repeatedly forgetting it was mid-detour and drifting
    # back into the same wall -- a memorylessness problem that a feedforward
    # policy structurally cannot fix, since it only ever sees the CURRENT
    # observation. Adding an LSTM (see Hausknecht & Stone, 2015, DRQN, in the
    # README) gives the policy a hidden state that persists across timesteps
    # within an episode, so it can carry information like "I'm currently
    # escaping a wall" forward instead of re-deciding from scratch every step.
    # policy_kwargs is left at sb3-contrib's defaults here (lstm_hidden_size=256,
    # n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True) -- see the
    # discussion of alternatives (hidden size, shared vs separate actor/critic
    # LSTM) before committing to this for the full curriculum.
    # n_epochs=10 (default) — learns quickly on fresh data
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./logs/",
        n_steps=n_steps,
        batch_size=200,
        n_epochs=10,
        device="cpu"  # avoid CPU/GPU non-determinism: never auto-select CUDA
    )

    model.learn(
        total_timesteps=12_000_000,
        tb_log_name="easy",
        callback=CallbackList(callbacks)
    )

    # Save the trained agent
    model.save("models/ppo_lstm_easy")

    if wandb_run is not None:
        wandb_run.finish()