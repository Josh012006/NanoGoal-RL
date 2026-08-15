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
        return Monitor(env.NanoEnv(difficulty="medium", worker_seed_offset=worker_idx))
    return _init


if __name__ == "__main__":
    # Automatically detect the number of available CPUs
    n_envs = min(os.cpu_count(), 8)
    n_steps = 6_000 // n_envs  # kept modest (vs. the old 20_000) to bound LSTM hidden-state
    # staleness -- see train_easy.py for the full rationale. Slightly larger than
    # easy's 4_000 to give medium's more complex episodes a bit more diversity per
    # update, while staying an order of magnitude below the old value.
    print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

    # SubprocVecEnv spawns one process per env, enabling true CPU parallelism
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Use RUN_ID from environment for unique checkpoint folder per run
    run_id = os.environ.get("RUN_ID", "local")
    checkpoint_path = f"./checkpoints/medium/{run_id}/"
    print(f"Checkpoint path: {checkpoint_path}")

    checkpoint_callback = KeepLastTwoCheckpoints(
        save_freq=1_000_000,
        save_path=checkpoint_path,
        name_prefix="ppo_medium"
    )

    # Tracks real per-seed training coverage (how many individual seeds from
    # each pool have actually been visited, and how often) — logged every
    # 100,000 timesteps to TensorBoard/WandB, with a final histogram per pool
    # saved to plots/medium/ once training ends.
    seed_coverage_callback = SeedCoverageCallback(
        log_freq=100_000,
        output_dir="plots/medium"
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
            name=f"medium_{run_id}",
            group=run_id,
            job_type="medium",
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

    # v3: RecurrentPPO/MultiInputLstmPolicy instead of plain PPO (see
    # train_easy.py for the full rationale). IMPORTANT: this .load() requires
    # models/ppo_lstm_easy to itself already be a RecurrentPPO checkpoint
    # produced by the new train_easy.py -- a v2-era plain-PPO checkpoint has
    # no LSTM weights and will fail to load here (architecture mismatch), so
    # easy must be retrained from scratch once before medium can chain off it.
    # n_epochs=8, batch_size=500: with n_steps*n_envs=6_000 transitions/rollout,
    # this gives 12 minibatches/epoch, so 8*12=96 total gradient steps taken on
    # a rollout before its LSTM hidden state is refreshed -- down from ~1_500
    # under the old n_steps=20_000/batch_size=200 (inherited)/n_epochs=15 config.
    # batch_size must be passed explicitly here (unlike n_steps/learning_rate/
    # n_epochs it was never overridden before, so it silently stayed at whatever
    # was pickled into ppo_lstm_easy -- 400 under the new train_easy.py).
    model = RecurrentPPO.load(
        "models/ppo_lstm_easy",
        env=vec_env,
        custom_objects={"n_steps": n_steps, "learning_rate": 1e-4, "n_epochs": 8, "batch_size": 500},
        device="cpu"  # avoid CPU/GPU non-determinism: never auto-select CUDA
    )

    model.learn(
        total_timesteps=200_000_000,
        reset_num_timesteps=False,
        tb_log_name="medium",
        callback=CallbackList(callbacks)
    )

    model.save("models/ppo_lstm_medium")

    if wandb_run is not None:
        wandb_run.finish()