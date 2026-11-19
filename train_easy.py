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

from stable_baselines3.common.env_checker import check_env
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
TOTAL_TIMESTEPS = 12_000_000
LR_START, LR_END = 3e-4, 5e-5


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
            difficulty="easy",
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
    args = parse_stage_args("easy", allow_init_from=False)

    # Check the environment first
    check_env(env.NanoEnv(difficulty="easy"))

    # Automatically detect the number of available CPUs
    n_envs = min(os.cpu_count(), 8)
    n_steps = 8_000 // n_envs  # kept modest (vs. the old 20_000) to bound LSTM hidden-state
    # staleness: RecurrentPPO reuses the SAME hidden state, captured once at rollout
    # time, across every PPO epoch on that rollout -- the more gradient steps taken
    # on a rollout before it's refreshed, the more that captured state drifts out of
    # sync with the (by-then-updated) weights that are supposed to have produced it.
    # 8_000 total transitions/rollout (with batch_size=2_000 below) keeps whole
    # episodes inside a single minibatch far more often than the old batch_size=200
    # did -- episodes cap at 800 steps (min(3 + 2*distance, 40s) / 0.05s timestep),
    # so a 200-400-sized minibatch was almost guaranteed to slice a long episode in
    # half mid-sequence, forcing a stale hidden state back in at that arbitrary cut.
    print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

    # ── Fresh run, or resume ──────────────────────────────────────────────────
    # With --resume <checkpoint folder> the run continues exactly where that
    # checkpoint stopped: same policy and optimizer, same timestep counter, the
    # curriculum (episode counters -> size of the three seed pools, sampling
    # streams, seed-visit counts, success statistics) and the learning-rate
    # schedule it was following (so the learning rate just carries on, and stays at
    # its final value if the run is extended past its budget).
    resume = load_resume(args.resume, "easy", n_envs) if args.resume else None
    if resume is not None:
        # Fail fast, with a readable message, if the checkpoint does not match the current
        # seed pools (otherwise the error would only surface inside a worker process).
        env.NanoEnv(difficulty="easy", curriculum_state=resume.worker_states[0]).close()
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
    checkpoint_path = f"./checkpoints/easy/{run_id}/"
    print(f"Checkpoint path: {checkpoint_path}")

    run_config = {
        "stage": "easy",
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
        name_prefix="ppo_easy",
        run_config=run_config,
        keep_all=True
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
    # n_epochs=8, batch_size=2_000: with n_steps*n_envs=8_000 transitions/rollout,
    # this gives 4 minibatches/epoch, so 8*4=32 total gradient steps taken on a
    # rollout before its LSTM hidden state is refreshed -- down from ~1_000 under
    # the old n_steps=20_000/batch_size=200/n_epochs=10 config. batch_size=2_000
    # (~2.5x the 800-step episode cap) was chosen over the theoretically-cleanest
    # fix (batch_size = full rollout, i.e. no minibatching at all, which would
    # fully eliminate mid-episode stale-state reinjection) because that measured
    # ~1.7GB peak RAM on this environment's 4GB VM and ran ~2.3x slower wall-clock
    # than batch_size=2_000 in a smoke test -- not worth it for a residual risk
    # that's already rare at this batch size, rather than the near-certainty it
    # was at batch_size=200-400.
    #
    # ent_coef=0.01, clip_range=0.1: a first easy run at the old hyperparameters
    # showed entropy collapsing steadily from step 0 (train/entropy_loss, which
    # is -entropy, rising from -2.8 to +3.7 over 17M steps -- i.e. actual entropy
    # falling continuously, unopposed, since ent_coef defaulted to 0.0) alongside
    # policy_gradient_loss and value_loss both blowing up after ~8-12M steps. A
    # near-deterministic Gaussian policy (very low action std) makes PPO's
    # probability ratios hypersensitive to small weight changes, which is a
    # well-documented general PPO late-training instability mode, independent of
    # the LSTM-specific staleness issue above. ent_coef=0.01 keeps some pressure
    # against entropy collapsing to that regime; clip_range=0.1 (down from the
    # 0.2 default) tightens the trust region as a second line of defense against
    # any single update (destabilized by either mechanism) moving too far.
    #
    # learning_rate: a LinearSchedule instead of a flat value, since the same run
    # trained fine for its first ~8M steps at a flat 3e-4 -- decaying from the
    # start would have slowed down learning that was already working, so this
    # only tapers off over the course of training rather than starting low.
    if resume is not None:
        # Everything else (ent_coef, clip_range, n_epochs, batch_size, ...) comes from the
        # checkpoint's policy.zip; only what depends on this machine/run is overridden.
        model = RecurrentPPO.load(
            resume.policy_path,
            env=vec_env,
            custom_objects={"n_steps": n_steps, "learning_rate": lr_schedule},
            device="cpu"  # avoid CPU/GPU non-determinism: never auto-select CUDA
        )
        prepare_model_for_resume(model, resume)
        report_resume(resume, model, lr_schedule, target_timesteps)
    else:
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log="./logs/",
            n_steps=n_steps,
            batch_size=2_000,
            n_epochs=8,
            ent_coef=0.01,
            clip_range=0.1,
            learning_rate=lr_schedule,
            device="cpu"  # avoid CPU/GPU non-determinism: never auto-select CUDA
        )

    # Fresh run: counters start at 0. Resume: the counter continues from the checkpoint
    # (reset_num_timesteps=False) and only the remaining steps are asked for, so the run
    # ends exactly at target_timesteps; TensorBoard keeps appending to the run folder the
    # checkpoint came from (logs/easy_N), so the curves continue.
    model.learn(
        total_timesteps=(target_timesteps - model.num_timesteps) if resume is not None else target_timesteps,
        reset_num_timesteps=resume is None,
        tb_log_name="easy",
        callback=CallbackList(callbacks)
    )

    # Save the trained agent
    model.save("models/ppo_lstm_easy")

    if wandb_run is not None:
        wandb_run.finish()