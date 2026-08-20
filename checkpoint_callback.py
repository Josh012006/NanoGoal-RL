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
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
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