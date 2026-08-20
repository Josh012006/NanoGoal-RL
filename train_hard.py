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
from stable_baselines3.common.utils import LinearSchedule
from sb3_contrib import RecurrentPPO
from checkpoint_callback import KeepLastNCheckpoints
from seed_coverage_callback import SeedCoverageCallback


def make_env(worker_idx):
    # worker_idx gives each parallel SubprocVecEnv worker a distinct
    # per-episode sampling stream (see env.py's worker_seed_offset), so
    # parallel workers explore genuinely different seeds instead of following
    # the same draw sequence in lockstep. Monitor is added manually here
    # (make_vec_env normally does this for us) since we build SubprocVecEnv
    # directly to be able to pass a different worker_idx to each instance.
    def _init():
        return Monitor(env.NanoEnv(difficulty="hard", worker_seed_offset=worker_idx))
    return _init


if __name__ == "__main__":
    # Automatically detect the number of available CPUs
    n_envs = min(os.cpu_count(), 8)
    n_steps = 16_000 // n_envs  # kept modest (vs. the old 20_000) to bound LSTM hidden-state
    # staleness -- see train_easy.py for the full rationale. Largest of the three
    # since hard's rollouts are the most expensive to collect, so it's worth
    # squeezing a bit more diversity per update even at a slightly higher
    # (but still bounded) staleness cost than easy/medium.
    print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

    # SubprocVecEnv spawns one process per env, enabling true CPU parallelism
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Use RUN_ID from environment for unique checkpoint folder per run
    run_id = os.environ.get("RUN_ID", "local")
    checkpoint_path = f"./checkpoints/hard/{run_id}/"
    print(f"Checkpoint path: {checkpoint_path}")

    checkpoint_callback = KeepLastNCheckpoints(
        save_freq=1_000_000,
        save_path=checkpoint_path,
        name_prefix="ppo_hard",
        keep_last_n=10
    )

    # Tracks real per-seed training coverage (how many individual seeds from
    # each pool have actually been visited, and how often) — logged every
    # 100,000 timesteps to TensorBoard/WandB, with a final histogram per pool
    # saved to plots/hard/ once training ends.
    seed_coverage_callback = SeedCoverageCallback(
        log_freq=100_000,
        output_dir="plots/hard"
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
            name=f"hard_{run_id}",
            group=run_id,
            job_type="hard",
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
    # train_easy.py for the full rationale) -- this is the actual fix being
    # tried for the hard-mode plateau/regression documented in the README's
    # "Final analysis". Same architecture-compatibility note as train_medium.py:
    # models/ppo_lstm_medium must already be a RecurrentPPO checkpoint.
    # n_epochs=10, batch_size=2_000: with n_steps*n_envs=16_000 transitions/rollout,
    # this gives 8 minibatches/epoch, so 10*8=80 total gradient steps taken on a
    # rollout before its LSTM hidden state is refreshed -- down from ~2_000 under
    # the old n_steps=20_000/batch_size=200 (inherited)/n_epochs=20 config. See
    # train_easy.py for why batch_size=2_000 rather than the full rollout.
    #
    # learning_rate: a fresh LinearSchedule continuing where medium's ended
    # (1e-5), decaying further over hard's own 400M-step budget -- see
    # train_easy.py for the general rationale (entropy collapse -> late-training
    # instability at a flat LR).
    model = RecurrentPPO.load(
        "models/ppo_lstm_medium",
        env=vec_env,
        custom_objects={
            "n_steps": n_steps,
            "learning_rate": LinearSchedule(start=1e-5, end=2e-6, end_fraction=1.0),
            "n_epochs": 10,
            "batch_size": 2_000,
        },
        device="cpu"  # avoid CPU/GPU non-determinism: never auto-select CUDA
    )

    model.learn(
        total_timesteps=400_000_000,
        reset_num_timesteps=False,
        tb_log_name="hard",
        callback=CallbackList(callbacks)
    )

    model.save("models/ppo_lstm_hard")

    if wandb_run is not None:
        wandb_run.finish()