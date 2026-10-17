"""Checkpoint bundles: a policy PLUS everything needed to resume a training.

A checkpoint written during training is a folder:

    checkpoints/<stage>/<RUN_ID>/ppo_<stage>_<timesteps>_steps/
        policy.zip         the policy, exactly what `model.save()` produces
        resume_state.json  the rest of the run's state (see `build_resume_state`)
        runtime_state.pkl  the episodes in progress, the LSTM states and the random
                           generators (see `build_runtime_state`); optional

`policy.zip` is a plain SB3 model: copy it to `models/ppo_lstm_<stage>.zip` (or
load it anywhere) and it behaves exactly like any other model. `models/` only
ever holds policies. `resume_state.json` is only read when RESUMING a training
(`train_<stage>.py --resume <checkpoint folder>`).

The resume state holds what is NOT in the policy zip and would otherwise be
lost: the curriculum (episode counters, hence the sizes of the easy / medium /
hard seed pools, the seed-drawing streams, the seed-visit counts and the
per-category success statistics of every worker), the learning rate and the
schedule it follows, the timestep target of the run, and a snapshot of the
hyper-parameters and library versions for sanity checks.

`runtime_state.pkl` is what makes a resume EXACT: it holds the episode every
worker was in the middle of (positions, clock, world seed, ...), the last
observation, the LSTM states of the policy and the global random generators, so
the resumed run goes on from the very same point instead of starting new
episodes. It is optional and only ever used when everything matches (same number
of workers, same timestep count, and the restored environments reproduce the
saved observations bit for bit); otherwise the resume falls back to starting new
episodes, with a message saying why.

This module is deliberately light (no torch / SB3 import at module level) so
that eval.py and visual_eval.py can use `resolve_policy_path` without cost.
"""
import argparse
import copy
import datetime
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np

FORMAT_VERSION = 1
POLICY_FILE = "policy.zip"
STATE_FILE = "resume_state.json"
RUNTIME_FILE = "runtime_state.pkl"
RUNTIME_VERSION = 1


# ── Locating a policy / a checkpoint ─────────────────────────────────────────

def is_checkpoint_dir(path) -> bool:
    """True if `path` is a checkpoint folder (it contains a policy.zip)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, POLICY_FILE))


def resolve_policy_path(spec) -> str:
    """Turns whatever the user points at into a path SB3's `load()` accepts.

    `spec` can be:
      - a checkpoint folder (checkpoints/medium/<RUN_ID>/ppo_medium_<N>_steps),
      - a policy .zip (models/ppo_lstm_medium.zip, a checkpoint's policy.zip, or a
        checkpoint of the old flat format ppo_medium_<N>_steps.zip),
      - the same without the .zip extension (as SB3 itself allows).
    Raises FileNotFoundError, listing what was tried, if nothing matches."""
    spec = os.fspath(spec).rstrip("/\\")
    candidates = []
    if os.path.isdir(spec):
        candidates.append(os.path.join(spec, POLICY_FILE))
    else:
        candidates.append(spec)
        if not spec.endswith(".zip"):
            candidates.append(spec + ".zip")
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No policy found for {spec!r}. Tried: {', '.join(candidates)}. "
        f"Pass a checkpoint folder, or the path of a policy .zip."
    )


def policy_label(spec) -> str:
    """Short, filesystem-safe name for a policy: the checkpoint folder's name
    (ppo_medium_20000000_steps) or the .zip's name without extension."""
    path = resolve_policy_path(spec)
    stem = os.path.basename(os.path.dirname(os.path.abspath(path))) if os.path.basename(path) == POLICY_FILE \
        else os.path.splitext(os.path.basename(path))[0]
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)


def _checkpoint_dir_of(spec) -> str:
    """Folder of the checkpoint `spec` refers to (a folder, or a file inside it)."""
    spec = os.fspath(spec).rstrip("/\\")
    if os.path.isfile(spec):
        spec = os.path.dirname(spec) or "."
    return spec


# ── Writing ──────────────────────────────────────────────────────────────────

def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("torch", "stable_baselines3", "sb3_contrib", "gymnasium"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def _hyperparameters(model) -> dict:
    """The hyper-parameters in force, for sanity checks at resume time (the
    authoritative copy lives inside policy.zip)."""
    out = {"n_envs": int(model.n_envs)}
    for name in ("n_steps", "batch_size", "n_epochs"):
        value = getattr(model, name, None)
        out[name] = int(value) if value is not None else None
    for name in ("ent_coef", "vf_coef", "max_grad_norm", "gamma", "gae_lambda"):
        value = getattr(model, name, None)
        out[name] = float(value) if value is not None else None
    clip_range = getattr(model, "clip_range", None)
    out["clip_range"] = float(clip_range(1.0)) if callable(clip_range) else (
        float(clip_range) if clip_range is not None else None)
    return out


def build_resume_state(model, vec_env, run_config: dict, final: bool = False) -> dict:
    """Collects the resume state of a running training (see the module docstring).

    `run_config` is provided by the training script: stage, run_id, target_timesteps
    (the absolute step the run aims for), lr_schedule (a StageLinearSchedule.spec())
    and fast_lstm (whether the single-call LSTM patch is active)."""
    workers = copy.deepcopy(vec_env.env_method("get_curriculum_state"))
    pool_sizes = {
        category: max(w["pool_sizes"][category] for w in workers)
        for category in ("easy", "medium", "hard")
    }
    return {
        "format_version": FORMAT_VERSION,
        "stage": run_config["stage"],
        "run_id": run_config.get("run_id"),
        "git_sha": _git_sha(),
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "final": bool(final),
        "num_timesteps": int(model.num_timesteps),
        "n_updates": int(model._n_updates),
        "target_timesteps": int(run_config["target_timesteps"]),
        "learning_rate": {
            # The value the optimizer is actually using at this point...
            "current": float(model.policy.optimizer.param_groups[0]["lr"]),
            # ...and the schedule that produces it as a function of the timestep counter.
            "schedule": run_config["lr_schedule"],
        },
        "curriculum": {
            "n_envs": len(workers),
            "pool_sizes": pool_sizes,                       # active size of each pool at this point
            "pool_totals": workers[0]["pool_totals"],       # size each pool grows to
            "workers": workers,
        },
        # The TensorBoard run folder this training writes to, so a resumed run appends to
        # the SAME curves (SB3 would otherwise pick the stage's latest folder, which is
        # not necessarily this run's).
        "tensorboard_dir": getattr(getattr(model, "logger", None), "dir", None),
        "training": _hyperparameters(model),
        "environment": {"fast_lstm": bool(run_config.get("fast_lstm", False)), **_versions()},
    }


def build_runtime_state(model, vec_env) -> dict:
    """The run's state in the MIDDLE of a rollout cycle, which is what an exact
    resume needs on top of policy.zip and resume_state.json: for each worker the
    episode in progress, plus the last observation / episode-start flags / LSTM
    states SB3 keeps between two rollouts, plus the global random generators
    (action sampling and minibatch shuffling).

    Must be called when the model and the environments are in step with each
    other, i.e. at a rollout boundary (start of a rollout or end of training)."""
    import torch as th

    if getattr(model, "_last_obs", None) is None:
        raise RuntimeError("the model has no current observation (training has not started)")
    states = model._last_lstm_states
    return {
        "version": RUNTIME_VERSION,
        "num_timesteps": int(model.num_timesteps),
        "n_envs": int(model.n_envs),
        "env_states": copy.deepcopy(vec_env.env_method("get_episode_state")),
        "last_obs": copy.deepcopy(model._last_obs),
        "last_episode_starts": np.array(model._last_episode_starts, copy=True),
        "lstm_states": {
            "pi": tuple(t.detach().cpu().numpy().copy() for t in states.pi),
            "vf": tuple(t.detach().cpu().numpy().copy() for t in states.vf),
        },
        "rng": {
            "torch": th.get_rng_state().numpy().copy(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }


def save_checkpoint(model, vec_env, save_dir: str, name: str, run_config: dict, final: bool = False) -> str:
    """Writes the checkpoint folder `save_dir/name` (policy.zip, resume_state.json
    and, when it can be collected, runtime_state.pkl).

    Written to a hidden temporary folder first and renamed at the end, so a
    process killed in the middle of a save (OOM killer, preemption) never leaves
    a half-written checkpoint under its final name. Returns the folder path."""
    final_dir = os.path.join(save_dir, name)
    tmp_dir = os.path.join(save_dir, ".tmp_" + name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    try:
        model.save(os.path.join(tmp_dir, "policy"))  # -> policy.zip
        try:
            state = build_resume_state(model, vec_env, run_config, final=final)
            # The episodes in progress / LSTM states: optional. If they cannot be collected
            # the checkpoint is still resumable, only not "exactly" (new episodes are started).
            state["runtime_state"] = None
            runtime_path = os.path.join(tmp_dir, RUNTIME_FILE)
            try:
                runtime = build_runtime_state(model, vec_env)
                with open(runtime_path, "wb") as f:
                    pickle.dump(runtime, f, protocol=4)
                state["runtime_state"] = {"file": RUNTIME_FILE, "version": RUNTIME_VERSION,
                                          "n_envs": runtime["n_envs"]}
            except Exception:
                if os.path.exists(runtime_path):
                    os.remove(runtime_path)
                print(f"WARNING: could not save the episodes in progress ({RUNTIME_FILE}); this checkpoint "
                      f"can still be resumed, but with new episodes:")
                traceback.print_exc()
            with open(os.path.join(tmp_dir, STATE_FILE), "w") as f:
                json.dump(state, f)
        except Exception:
            # A problem collecting the extra state must never cost the policy:
            # keep the bundle with the policy only (it just cannot be resumed).
            print("WARNING: could not write resume_state.json, this checkpoint holds the policy only:")
            traceback.print_exc()
        shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return final_dir


# ── Reading / resuming ───────────────────────────────────────────────────────

def remap_worker_states(saved_workers: list, n_envs: int) -> list:
    """One curriculum state per worker of the NEW run.

    Same number of workers as when the checkpoint was written: used as is.
    Different number (e.g. resuming on a machine with another CPU count): every
    worker gets the average episode counter (the pools' size depends on the
    per-worker count, so the pools stay where they were), workers that existed
    keep their own sampling stream, new workers keep their fresh one, and the
    visit counts / success statistics of workers that no longer exist are folded
    into worker 0 so the coverage and success totals are preserved."""
    if len(saved_workers) == n_envs:
        return copy.deepcopy(saved_workers)

    mean_episodes = int(round(sum(w["episodes"] for w in saved_workers) / len(saved_workers)))
    template = saved_workers[0]
    out = []
    for i in range(n_envs):
        if i < len(saved_workers):
            worker = copy.deepcopy(saved_workers[i])
        else:
            worker = {
                "difficulty": template["difficulty"],
                "pool_fingerprint": template["pool_fingerprint"],
                "sampling_rng_state": None,
                "seed_counts": {"easy": {}, "medium": {}, "hard": {}},
                "category_stats": {c: {"episodes": 0, "successes": 0} for c in ("easy", "medium", "hard")},
            }
        worker["episodes"] = mean_episodes
        out.append(worker)

    for dropped in saved_workers[n_envs:]:
        for category in ("easy", "medium", "hard"):
            counts = out[0]["seed_counts"].setdefault(category, {})
            for seed, count in dropped["seed_counts"].get(category, {}).items():
                counts[seed] = counts.get(seed, 0) + count
            for key in ("episodes", "successes"):
                out[0]["category_stats"][category][key] += dropped["category_stats"][category][key]
    return out


@dataclass
class ResumeInfo:
    checkpoint_dir: str
    policy_path: str
    state: dict
    worker_states: list         # one curriculum state per worker of the new run
    target_timesteps: int       # absolute step the run aims for (can be overridden by the user)
    lr_schedule_spec: dict      # the schedule the checkpointed run was following
    num_timesteps: int
    tensorboard_dir: Optional[str] = None   # run folder to keep appending to (None: SB3's default choice)
    runtime: Optional[dict] = None          # runtime_state.pkl if it can be used (exact resume), else None
    runtime_note: str = ""                  # why `runtime` is None (printed by report_resume)
    continuation: str = "restarted"         # "exact" or "restarted"; set by prepare_model_for_resume
    continuation_note: str = ""


def load_resume(spec, stage: str, n_envs: int) -> ResumeInfo:
    """Reads a checkpoint folder for resuming. Raises a clear error if it cannot be resumed."""
    ckpt_dir = _checkpoint_dir_of(spec)
    state_path = os.path.join(ckpt_dir, STATE_FILE)
    policy_path = os.path.join(ckpt_dir, POLICY_FILE)
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(
            f"{ckpt_dir!r} is not a checkpoint folder (no {POLICY_FILE} inside). "
            f"Resume from a folder such as checkpoints/{stage}/<RUN_ID>/ppo_{stage}_<N>_steps."
        )
    if not os.path.isfile(state_path):
        raise FileNotFoundError(
            f"{ckpt_dir!r} has no {STATE_FILE}: it cannot be RESUMED (it was written before "
            f"resumable checkpoints existed, or its state could not be saved). Its policy is still "
            f"usable as a starting point with --init-from."
        )
    with open(state_path) as f:
        state = json.load(f)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported resume-state format {state.get('format_version')!r} (expected {FORMAT_VERSION}).")
    if state.get("stage") != stage:
        raise ValueError(f"This checkpoint belongs to the {state.get('stage')!r} stage, not {stage!r}.")

    runtime, runtime_note = _load_runtime(ckpt_dir, state, n_envs)
    return ResumeInfo(
        checkpoint_dir=ckpt_dir,
        policy_path=policy_path,
        state=state,
        worker_states=remap_worker_states(state["curriculum"]["workers"], n_envs),
        target_timesteps=int(state["target_timesteps"]),
        lr_schedule_spec=state["learning_rate"]["schedule"],
        num_timesteps=int(state["num_timesteps"]),
        tensorboard_dir=state.get("tensorboard_dir"),
        runtime=runtime,
        runtime_note=runtime_note,
    )


def _load_runtime(ckpt_dir: str, state: dict, n_envs: int):
    """Reads runtime_state.pkl if the checkpoint has one that can be used with this
    run. Returns (runtime or None, reason it is None). Never raises: whatever is wrong,
    the resume simply starts new episodes."""
    info = state.get("runtime_state")
    if not info:
        return None, "this checkpoint has no saved episodes in progress"
    try:
        path = os.path.join(ckpt_dir, info.get("file", RUNTIME_FILE))
        with open(path, "rb") as f:
            runtime = pickle.load(f)
    except Exception as exc:
        return None, f"{RUNTIME_FILE} could not be read ({type(exc).__name__}: {exc})"
    if runtime.get("version") != RUNTIME_VERSION:
        return None, f"{RUNTIME_FILE} has format version {runtime.get('version')!r}, expected {RUNTIME_VERSION}"
    if runtime.get("num_timesteps") != state.get("num_timesteps"):
        return None, f"{RUNTIME_FILE} is at {runtime.get('num_timesteps')} timesteps, resume_state.json at {state.get('num_timesteps')}"
    if runtime.get("n_envs") != n_envs:
        return None, (f"the checkpoint was written with {runtime.get('n_envs')} workers and this run has {n_envs} "
                      f"(an episode in progress belongs to a given worker)")
    return runtime, ""


def _restore_runtime(model, runtime: dict) -> None:
    """Puts the environments back in the middle of their saved episodes and restores
    what SB3 keeps between two rollouts, so `learn(reset_num_timesteps=False)` goes on
    exactly where the checkpointed run was. Raises if anything does not match; the caller
    then falls back to new episodes (which overwrites whatever was applied here)."""
    import torch as th
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    vec_env = model.get_env()
    n_envs = model.n_envs
    saved_obs = runtime["last_obs"]

    # 1. The episodes in progress. Each restored environment must reproduce, bit for bit,
    #    the observation the policy saw at that point: this checks the whole chain (world
    #    rebuilt from its seed, positions, orientation, lidar...).
    for i in range(n_envs):
        obs = vec_env.env_method("set_episode_state", runtime["env_states"][i], indices=[i])[0]
        for key, value in obs.items():
            if not np.array_equal(np.asarray(value), np.asarray(saved_obs[key][i])):
                raise RuntimeError(f"worker {i}: restored observation {key!r} differs from the saved one")

    # 2. What SB3 carries from one rollout to the next.
    model._last_obs = copy.deepcopy(saved_obs)
    model._last_episode_starts = np.array(runtime["last_episode_starts"], copy=True)
    lstm = runtime["lstm_states"]
    model._last_lstm_states = RNNStates(
        pi=tuple(th.as_tensor(a, dtype=th.float32, device=model.device).clone() for a in lstm["pi"]),
        vf=tuple(th.as_tensor(a, dtype=th.float32, device=model.device).clone() for a in lstm["vf"]),
    )

    # 3. Random generators, last (nothing above draws random numbers).
    rng = runtime["rng"]
    th.set_rng_state(th.from_numpy(np.array(rng["torch"], dtype=np.uint8)))
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def prepare_model_for_resume(model, resume: Optional[ResumeInfo] = None) -> None:
    """Call on a model loaded from a checkpoint, right before
    `learn(reset_num_timesteps=False)`.

    SB3 pickles the last observation and the LSTM states of the run that saved
    the model. With `reset_num_timesteps=False` it would then skip resetting the
    new, freshly built environments and feed them that stale observation.

    Two ways out, recorded in `resume.continuation` (printed by `report_resume`):
      - "exact": the checkpoint has runtime_state.pkl and it fits this run. The
        environments are put back in the middle of the episodes that were running,
        and the LSTM states, last observation and random generators are restored:
        the run goes on from the very same point.
      - "restarted": otherwise (no runtime state, another number of workers, a
        mismatch...). `_last_obs` is cleared so `learn()` resets the environments
        (every worker starts a new episode, with the restored curriculum) and the
        LSTM states restart from zero like at the start of any episode.

    When `resume` is given and records a TensorBoard folder, the model is also pointed
    back at that folder: with `reset_num_timesteps=False` SB3 would otherwise append to
    the stage's LATEST run folder (logs/<stage>_N), which is a different run if
    several runs of the stage exist."""
    import torch as th
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    from stable_baselines3.common.logger import configure

    if resume is not None and resume.tensorboard_dir:
        model.set_logger(configure(resume.tensorboard_dir, ["stdout", "tensorboard"]))

    note = resume.runtime_note if resume is not None else ""
    if resume is not None and resume.runtime is not None:
        try:
            _restore_runtime(model, resume.runtime)
            resume.continuation, resume.continuation_note = "exact", ""
            return
        except Exception as exc:
            note = f"the saved episodes could not be restored ({type(exc).__name__}: {exc})"
            print(f"WARNING: {note}; new episodes will be started instead.")
            traceback.print_exc()

    model._last_obs = None
    model._last_episode_starts = np.ones((model.n_envs,), dtype=bool)
    states = model._last_lstm_states
    model._last_lstm_states = RNNStates(
        pi=tuple(th.zeros_like(t) for t in states.pi),
        vf=tuple(th.zeros_like(t) for t in states.vf),
    )
    if resume is not None:
        resume.continuation, resume.continuation_note = "restarted", note


def report_resume(resume: ResumeInfo, model, lr_schedule, target_timesteps: int) -> None:
    """Sanity checks right after a checkpoint has been loaded for resuming, and a
    summary of where the run picks up. Raises if the policy and the resume state do
    not belong together; only warns about a changed configuration."""
    state = resume.state
    if int(model.num_timesteps) != resume.num_timesteps:
        raise RuntimeError(
            f"policy.zip is at {model.num_timesteps} timesteps but resume_state.json says "
            f"{resume.num_timesteps}: the two files of this checkpoint do not match."
        )

    # The schedule is a function of the absolute step, so evaluating it at the resumed
    # step must give back the learning rate the optimizer was using when saved.
    lr_now = float(lr_schedule(1.0 - model.num_timesteps / target_timesteps))
    lr_saved = float(state["learning_rate"]["current"])
    lr_ok = abs(lr_now - lr_saved) <= 1e-6 * max(abs(lr_saved), 1e-12)

    print("──────── Resuming ────────")
    print(f"Checkpoint        : {resume.checkpoint_dir} (saved {state.get('saved_at')}, run {state.get('run_id')})")
    print(f"Timesteps         : {model.num_timesteps:,} -> target {target_timesteps:,} "
          f"({target_timesteps - model.num_timesteps:,} to go)")
    print(f"Learning rate     : {lr_now:.3e} now (checkpoint: {lr_saved:.3e}), schedule "
          f"{state['learning_rate']['schedule']['start']:.1e} -> {state['learning_rate']['schedule']['end']:.1e} "
          f"over {state['learning_rate']['schedule']['budget']:,} steps")
    pools = state["curriculum"]["pool_sizes"]
    totals = state["curriculum"]["pool_totals"]
    print("Seed pools        : " + ", ".join(f"{c} {pools[c]}/{totals[c]}" for c in ("easy", "medium", "hard")))
    saved_workers = state["curriculum"]["n_envs"]
    print(f"Workers           : {saved_workers} in the checkpoint, {len(resume.worker_states)} now")
    print(f"TensorBoard       : {resume.tensorboard_dir or 'latest run folder of the stage'} (curves continue)")
    if resume.continuation == "exact":
        print("Episodes          : the episodes in progress, LSTM states and random generators are restored "
              "(the run goes on from the very same point)")
    else:
        print("Episodes          : new episodes are started (LSTM states from zero)"
              + (f" because {resume.continuation_note}" if resume.continuation_note else ""))
    if not lr_ok:
        print(f"WARNING: the learning rate at this step ({lr_now:.6e}) differs from the one saved in the "
              f"checkpoint ({lr_saved:.6e}).")

    saved_hp = state.get("training", {})
    now_hp = _hyperparameters(model)
    for key in ("n_envs", "n_steps", "batch_size", "n_epochs", "ent_coef", "clip_range", "gamma", "gae_lambda"):
        if key in saved_hp and saved_hp[key] != now_hp.get(key) and not (
                isinstance(saved_hp[key], float) and abs(saved_hp[key] - now_hp[key]) < 1e-12):
            print(f"WARNING: {key} was {saved_hp[key]} in the checkpointed run and is {now_hp.get(key)} now.")
    print("──────────────────────────")


# ── Command line shared by the three training scripts ────────────────────────

def parse_stage_args(stage: str, allow_init_from: bool = True, argv=None):
    """`--resume`, `--init-from` and `--total-timesteps` for train_<stage>.py.

    --resume and --init-from can also be given through the RESUME_FROM and
    INIT_FROM environment variables (used by the CI's train.sh); an empty value
    means "not set". Everything is optional: with no argument the script behaves
    exactly like a fresh run."""
    parser = argparse.ArgumentParser(description=f"Train the {stage} stage.")
    parser.add_argument(
        "--resume", default=os.environ.get("RESUME_FROM") or None, metavar="CHECKPOINT",
        help="Resume this run from a checkpoint folder (checkpoints/<stage>/<RUN_ID>/ppo_<stage>_<N>_steps): "
             "same policy, optimizer, curriculum pools, learning-rate schedule and timestep counter. "
             "The learning rate keeps following its schedule (and stays at its final value past the end of it).",
    )
    if allow_init_from:
        parser.add_argument(
            "--init-from", default=os.environ.get("INIT_FROM") or None, metavar="POLICY",
            help="Start a FRESH run of this stage from this policy instead of the previous stage's "
                 "models/ppo_lstm_<stage>.zip: a policy .zip or a checkpoint folder. "
                 "Counters, curriculum and learning rate start from scratch.",
        )
    parser.add_argument(
        "--total-timesteps", type=int, default=None, metavar="N",
        help="Absolute timestep the run aims for. Default: the script's TOTAL_TIMESTEPS (fresh run) or the "
             "checkpoint's target (resume). On a fresh run the learning-rate schedule is stretched over it; "
             "on a resume it extends the run past its original budget, the learning rate staying at the "
             "schedule's final value beyond it.",
    )
    args = parser.parse_args(argv)
    if args.resume and getattr(args, "init_from", None):
        parser.error("--resume and --init-from are mutually exclusive: --resume continues a run, "
                     "--init-from starts a new one from a policy.")
    if args.total_timesteps is not None and args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive.")
    if not allow_init_from:
        args.init_from = None
    return args
