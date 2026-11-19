import os
import glob
import shutil
import traceback
from stable_baselines3.common.callbacks import BaseCallback

from .checkpoints import save_checkpoint, POLICY_FILE


# Printed (once per checkpoint) when keep_all is on and the disk is getting full.
LOW_DISK_WARNING_GB = 5


class ResumableCheckpointCallback(BaseCallback):
    """Writes a resumable checkpoint every `save_freq` timesteps (and one when
    training ends). Each checkpoint is a folder holding the policy
    (`policy.zip`, a regular SB3 model) next to `resume_state.json`; see
    checkpoints.py for the format and for how to resume from one.

    Args:
        save_freq:     a checkpoint is written the first time the timestep counter has
                       crossed a new multiple of this value (see _on_rollout_start).
        save_path:     folder receiving the checkpoint folders.
        name_prefix:   checkpoints are named <name_prefix>_<timesteps>_steps.
        run_config:    {"stage", "run_id", "target_timesteps", "lr_schedule", "fast_lstm"},
                       stored in the resume state (see checkpoints.build_resume_state).
        keep_all:      keep every checkpoint (they are ~7 MB each). When False, only the
                       `keep_last_n` most recent ones are kept.
        keep_last_n:   how many checkpoints to keep when keep_all is False.
    """

    def __init__(self, save_freq, save_path, name_prefix, run_config,
                 keep_all=False, keep_last_n=10, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.run_config = run_config
        self.keep_all = keep_all
        self.keep_last_n = keep_last_n
        self._last_saved_milestone = 0
        self._last_saved_steps = None
        os.makedirs(save_path, exist_ok=True)
        # A save interrupted by a kill leaves a hidden .tmp_* folder behind; it is
        # never read by anything, so just clear it.
        for leftover in glob.glob(os.path.join(save_path, ".tmp_*")):
            shutil.rmtree(leftover, ignore_errors=True)

    def _on_training_start(self):
        # On a resumed run the counter starts far from 0: do not re-save at once the
        # milestone the previous run had already reached.
        self._last_saved_milestone = self.model.num_timesteps // self.save_freq

    def _save(self, final=False):
        steps = int(self.model.num_timesteps)
        name = f"{self.name_prefix}_{steps}_steps"
        # A checkpoint that cannot be written (disk full, ...) is reported loudly but must
        # not kill a run that may have been going for days: training carries on and the
        # next milestone tries again.
        try:
            path = save_checkpoint(self.model, self.training_env, self.save_path, name,
                                   self.run_config, final=final)
        except Exception:
            print(f"WARNING: checkpoint {name} could NOT be written, training continues:")
            traceback.print_exc()
            return
        self._last_saved_steps = steps
        if self.verbose:
            print(f"Checkpoint saved: {path}")
        if not self.keep_all:
            self._prune()
        else:
            self._warn_if_disk_low()

    def _warn_if_disk_low(self):
        # With keep_all every checkpoint stays on disk (~7 MB each): say so well before it fills up.
        try:
            free_gb = shutil.disk_usage(self.save_path).free / 1e9
        except OSError:
            return
        if free_gb < LOW_DISK_WARNING_GB:
            print(f"WARNING: only {free_gb:.1f} GB of disk left where checkpoints are written "
                  f"({self.save_path}); delete old checkpoints or runs before it fills up.")

    def _prune(self):
        checkpoints = sorted(
            (d for d in glob.glob(os.path.join(self.save_path, f"{self.name_prefix}_*_steps"))
             if os.path.isfile(os.path.join(d, POLICY_FILE))),
            key=os.path.getmtime,
        )
        for old in checkpoints[:-self.keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)
            if self.verbose:
                print(f"Deleted old checkpoint: {old}")

    def _on_rollout_start(self):
        # Checkpoints are taken at the START of a rollout, i.e. right after the
        # previous PPO update: the policy and optimizer are fully up to date, the
        # timestep counter sits on a rollout boundary, and no half-collected rollout
        # is lost on resume. (A crossed-milestone check on num_timesteps, so it fires
        # every save_freq TOTAL timesteps whatever the number of parallel envs.)
        current_milestone = self.model.num_timesteps // self.save_freq
        if current_milestone > self._last_saved_milestone:
            self._last_saved_milestone = current_milestone
            self._save()

    def _on_step(self):
        return True

    def _on_training_end(self):
        # The state at the very end of training, so a finished run can be extended.
        if self._last_saved_steps != int(self.model.num_timesteps):
            self._save(final=True)
