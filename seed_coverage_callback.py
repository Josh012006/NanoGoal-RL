"""
seed_coverage_callback.py

Tracks how many INDIVIDUAL seeds from the easy/medium/hard training pools
have actually been visited during training, and how many times each. This
gives real visibility into training coverage -- two runs with the same
timestep count could have wildly different seed diversity depending on
episode length, and this makes that visible instead of assumed.

Each parallel sub-environment keeps its own {seed: times_drawn} dict in
memory (see env.py's _seed_counts / get_seed_visit_counts). This callback
periodically pulls those dicts from every sub-environment via
VecEnv.env_method() (works transparently across SubprocVecEnv's separate
processes, no shared files, no race conditions) and merges them.

Logs, every `log_freq` timesteps, to TensorBoard/WandB:
    seed_coverage/{easy,medium,hard}_unique_seen
    seed_coverage/{easy,medium,hard}_pct_unique_seen

At the end of training, saves one histogram per pool (as a PNG) showing the
distribution of how many times each individual seed was visited over the
whole run -- useful to spot whether the curriculum's pool-expansion schedule
is leaving some seeds under-sampled while others are seen far more often.
"""
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class SeedCoverageCallback(BaseCallback):
    def __init__(self, log_freq=100_000, output_dir="plots", verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.output_dir = output_dir
        self._pool_sizes = None      # {"easy": n, "medium": n, "hard": n} -- fetched once
        self._last_milestone = 0     # last num_timesteps // log_freq value we logged at

    def _on_training_start(self) -> None:
        # Pool sizes are identical across every sub-environment (same
        # seeds.json, same fixed _episode_rng seed), so querying just one
        # sub-env (index 0) is enough -- no need to aggregate this part.
        sizes_per_env = self.training_env.env_method("get_training_pool_sizes", indices=[0])
        self._pool_sizes = sizes_per_env[0]

    def _aggregate_counts(self):
        """Merges every sub-env's {category: {seed: count}} into one combined
        dict covering the whole training run so far."""
        per_env_counts = self.training_env.env_method("get_seed_visit_counts")
        combined = {"easy": {}, "medium": {}, "hard": {}}
        for env_counts in per_env_counts:
            for category, seed_counts in env_counts.items():
                for seed, count in seed_counts.items():
                    combined[category][seed] = combined[category].get(seed, 0) + count
        return combined

    def _on_step(self) -> bool:
        # Using num_timesteps // log_freq (a crossed-milestone check) instead
        # of a plain modulo so this triggers reliably every log_freq TOTAL
        # timesteps regardless of how many parallel envs are running (with
        # SubprocVecEnv, num_timesteps jumps by n_envs each call, which could
        # step over an exact modulo boundary depending on n_envs).
        current_milestone = self.num_timesteps // self.log_freq
        if current_milestone > self._last_milestone:
            self._last_milestone = current_milestone
            self._log_coverage()
        return True

    def _log_coverage(self):
        combined = self._aggregate_counts()
        for category, pool_size in self._pool_sizes.items():
            n_unique = len(combined[category])
            pct = (n_unique / pool_size * 100) if pool_size > 0 else 0.0
            self.logger.record(f"seed_coverage/{category}_unique_seen", n_unique)
            self.logger.record(f"seed_coverage/{category}_pct_unique_seen", pct)
        # Explicit dump so these scalars are written to TensorBoard/WandB
        # right away, rather than waiting for PPO's own next internal flush.
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            parts = [f"{c}={len(combined[c])}/{self._pool_sizes[c]}" for c in ("easy", "medium", "hard")]
            print(f"[SeedCoverageCallback] step {self.num_timesteps}: " + "  ".join(parts))

    def _on_training_end(self) -> None:
        combined = self._aggregate_counts()
        os.makedirs(self.output_dir, exist_ok=True)

        for category, seed_counts in combined.items():
            if not seed_counts:
                continue
            pool_size = self._pool_sizes.get(category, 0)
            n_unique = len(seed_counts)
            pct = (n_unique / pool_size * 100) if pool_size > 0 else 0.0
            counts = list(seed_counts.values())

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(counts, bins=30, color="#378ADD", edgecolor="#185FA5", alpha=0.85)
            ax.set_xlabel("Times an individual seed was visited")
            ax.set_ylabel("Number of distinct seeds")
            ax.set_title(
                f"Seed visit distribution — {category} pool "
                f"({n_unique}/{pool_size} seeds seen, {pct:.1f}% coverage)"
            )
            plt.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"seed_coverage_{category}.png"), dpi=150)
            plt.close(fig)

            if self.verbose:
                print(f"[SeedCoverageCallback] {category}: {n_unique}/{pool_size} "
                      f"unique seeds seen ({pct:.1f}%), saved histogram to {self.output_dir}.")