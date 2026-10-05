import os
import glob
from stable_baselines3.common.callbacks import BaseCallback


class KeepLastNCheckpoints(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, keep_last_n=10, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.keep_last_n = keep_last_n
        self._last_saved_milestone = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        # Using num_timesteps // save_freq (a crossed-milestone check) instead
        # of a plain modulo on n_calls, so this triggers reliably every
        # save_freq TOTAL timesteps regardless of how many parallel envs are
        # running -- with SubprocVecEnv, n_calls only advances by 1 per
        # rollout step while num_timesteps advances by n_envs, so comparing
        # against n_calls silently saved every save_freq * n_envs timesteps
        # instead (e.g. every 2M instead of 1M with n_envs=2). Matches the
        # same fix already applied to SeedCoverageCallback.
        current_milestone = self.num_timesteps // self.save_freq
        if current_milestone > self._last_saved_milestone:
            self._last_saved_milestone = current_milestone

            # Sauvegarder le nouveau checkpoint
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose:
                print(f"Checkpoint saved: {path}")

            # Garder seulement les self.keep_last_n derniers
            checkpoints = sorted(
                glob.glob(os.path.join(self.save_path, f"{self.name_prefix}_*.zip")),
                key=os.path.getmtime
            )
            for old in checkpoints[:-self.keep_last_n]:
                os.remove(old)
                if self.verbose:
                    print(f"Deleted old checkpoint: {old}")

        return True