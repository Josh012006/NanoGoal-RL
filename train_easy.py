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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints


def make_env():
    return env.NanoEnv(difficulty="easy")


if __name__ == "__main__":
    # Check the environment first
    check_env(env.NanoEnv(difficulty="easy"))

    # Automatically detect the number of available CPUs
    n_envs = min(os.cpu_count(), 8)
    n_steps = 20_000 // n_envs  # increased to reduce backprop proportion vs rollout
    print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

    # SubprocVecEnv spawns one process per env, enabling true CPU parallelism
    vec_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )

    # Use RUN_ID from environment for unique checkpoint folder per run
    run_id = os.environ.get("RUN_ID", "local")
    checkpoint_path = f"./checkpoints/easy/{run_id}/"
    print(f"Checkpoint path: {checkpoint_path}")

    checkpoint_callback = KeepLastTwoCheckpoints(
        save_freq=1_000_000,
        save_path=checkpoint_path,
        name_prefix="ppo_easy"
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

    callbacks = [checkpoint_callback]
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    # Define and train the agent
    # n_epochs=10 (default) — learns quickly on fresh data
    model = PPO(
        "MultiInputPolicy",
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
    model.save("models/ppo_nanogoal_easy")

    if wandb_run is not None:
        wandb_run.finish()