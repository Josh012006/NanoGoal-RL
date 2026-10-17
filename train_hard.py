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

from nanogoal_rl import env

from stable_baselines3.common.vec_env import SubprocVecEnv
from nanogoal_rl.monitor import ResumableMonitor
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib import RecurrentPPO
from nanogoal_rl.checkpoint_callback import ResumableCheckpointCallback
from nanogoal_rl.checkpoints import (
    parse_stage_args, load_resume, prepare_model_for_resume, report_resume, resolve_policy_path
)
from nanogoal_rl.schedules import StageLinearSchedule
from nanogoal_rl.seed_coverage_callback import SeedCoverageCallback
from nanogoal_rl import fast_lstm


# Budget of this stage, counted on its own timestep counter. The learning rate decays
# linearly from LR_START to LR_END over exactly this many timesteps (then stays at
# LR_END, see nanogoal_rl/schedules.py) and the run stops there, unless
# --total-timesteps says otherwise. Raise TOTAL_TIMESTEPS to give a fresh run more steps.
TOTAL_TIMESTEPS = 400_000_000
LR_START, LR_END = 1e-5, 2e-6


def make_env(worker_idx, curriculum_state=None):
    # worker_idx gives each parallel SubprocVecEnv worker a distinct
    # per-episode sampling stream (see env.py's worker_seed_offset), so
    # parallel workers explore genuinely different seeds instead of following
    # the same draw sequence in lockstep. The monitor is added manually here
    # (make_vec_env normally does this for us) since we build SubprocVecEnv
    # directly to be able to pass a different worker_idx to each instance.
    # ResumableMonitor is SB3's Monitor that can also save and restore the
    # episode in progress (exact resume, see nanogoal_rl/checkpoints.py).
    # curriculum_state (None for a fresh run) restores a checkpoint's curriculum
    # for this worker: episode counter (pool sizes), sampling stream, visit counts.
    def _init():
        return ResumableMonitor(env.NanoEnv(
            difficulty="hard",
            worker_seed_offset=worker_idx,
            curriculum_state=curriculum_state
        ))
    return _init


if __name__ == "__main__":
    # Replace sb3-contrib's per-timestep Python LSTM loop (the bottleneck of every
    # PPO update) by a single, numerically equivalent LSTM call. Skipped with a
    # warning if the installed sb3-contrib is not the tested one. Must run before
    # the model is created or loaded; done here (not at import time) so that
    # SubprocVecEnv workers do not repeat it.
    fast_lstm_active = fast_lstm.apply()

    # Optional: --resume <checkpoint folder> / --init-from <policy> / --total-timesteps N
    # (see nanogoal_rl/checkpoints.py). No argument = a fresh run, as before.
    args = parse_stage_args("hard")

    # Automatically detect the number of available CPUs
    n_envs = min(os.cpu_count(), 8)
    n_steps = 16_000 // n_envs  # kept modest (vs. the old 20_000) to bound LSTM hidden-state
    # staleness -- see train_easy.py for the full rationale. Largest of the three
    # since hard's rollouts are the most expensive to collect, so it's worth
    # squeezing a bit more diversity per update even at a slightly higher
    # (but still bounded) staleness cost than easy/medium.
    print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

    # ── Fresh run, or resume ──────────────────────────────────────────────────
    # With --resume <checkpoint folder> the run continues exactly where that
    # checkpoint stopped: same policy and optimizer, same timestep counter, the
    # curriculum (episode counters -> size of the three seed pools, sampling
    # streams, seed-visit counts, success statistics) and the learning-rate
    # schedule it was following (so the learning rate just carries on, and stays at
    # its final value if the run is extended past its budget).
    resume = load_resume(args.resume, "hard", n_envs) if args.resume else None
    if resume is not None:
        # Fail fast, with a readable message, if the checkpoint does not match the current
        # seed pools (otherwise the error would only surface inside a worker process).
        env.NanoEnv(difficulty="hard", curriculum_state=resume.worker_states[0]).close()
        target_timesteps = args.total_timesteps or resume.target_timesteps
        if target_timesteps <= resume.num_timesteps:
            raise SystemExit(
                f"Nothing to train: the checkpoint is already at {resume.num_timesteps:,} timesteps "
                f"and the target is {target_timesteps:,}. Pass a larger --total-timesteps to extend the run."
            )
        lr_schedule = StageLinearSchedule.from_spec(resume.lr_schedule_spec, session_end=target_timesteps)
    else:
        target_timesteps = args.total_timesteps or TOTAL_TIMESTEPS
        lr_schedule = StageLinearSchedule(LR_START, LR_END, budget=target_timesteps)

    # SubprocVecEnv spawns one process per env, enabling true CPU parallelism
    vec_env = SubprocVecEnv([
        make_env(i, resume.worker_states[i] if resume is not None else None)
        for i in range(n_envs)
    ])

    # Use RUN_ID from environment for unique checkpoint folder per run
    run_id = os.environ.get("RUN_ID", "local")
    checkpoint_path = f"./checkpoints/hard/{run_id}/"
    print(f"Checkpoint path: {checkpoint_path}")

    run_config = {
        "stage": "hard",
        "run_id": run_id,
        "target_timesteps": target_timesteps,
        "lr_schedule": lr_schedule.spec(),
        "fast_lstm": fast_lstm_active,
    }

    # Every checkpoint is kept (keep_all=True; ~7 MB each, the folder is git-ignored)
    # and is a folder holding the policy (policy.zip, a regular SB3 model) next to the
    # state needed to resume (resume_state.json). Set keep_all=False, keep_last_n=N to
    # go back to keeping only the N most recent ones.
    checkpoint_callback = ResumableCheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_path,
        name_prefix="ppo_hard",
        run_config=run_config,
        keep_all=True
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
    # Starting policy: the checkpoint's own when resuming, otherwise --init-from if given
    # (a policy .zip or a checkpoint folder), otherwise the previous stage's model.
    if resume is not None:
        start_policy = resume.policy_path
    elif args.init_from:
        start_policy = resolve_policy_path(args.init_from)
    else:
        start_policy = "models/ppo_lstm_medium"
    print(f"Starting policy: {start_policy}")
    model = RecurrentPPO.load(
        start_policy,
        env=vec_env,
        custom_objects={
            "n_steps": n_steps,
            "learning_rate": lr_schedule,
            "n_epochs": 10,
            "batch_size": 2_000,
        },
        device="cpu"  # avoid CPU/GPU non-determinism: never auto-select CUDA
    )
    if resume is not None:
        prepare_model_for_resume(model, resume)
        report_resume(resume, model, lr_schedule, target_timesteps)

    # reset_num_timesteps=True: hard's own step counter starts at 0 instead of
    # inheriting the steps of the loaded ppo_lstm_medium (see train_medium.py).
    # TensorBoard's x-axis, the checkpoint names (ppo_hard_<steps>_steps) and
    # both callbacks' 100k milestones are relative to hard's own 400M budget,
    # and the learning-rate schedule above runs over its whole 1e-5 -> 2e-6 range. The
    # run is written to logs/hard_1.
    # (On a resume the counter instead continues from the checkpoint, only the remaining
    # steps are asked for so the run ends exactly at target_timesteps, and TensorBoard
    # keeps appending to the run folder the checkpoint came from so the curves continue.)
    model.learn(
        total_timesteps=(target_timesteps - model.num_timesteps) if resume is not None else target_timesteps,
        reset_num_timesteps=resume is None,
        tb_log_name="hard",
        callback=CallbackList(callbacks)
    )

    model.save("models/ppo_lstm_hard")

    if wandb_run is not None:
        wandb_run.finish()