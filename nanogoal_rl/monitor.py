"""Monitor wrapper that can save and restore the episode in progress.

`ResumableMonitor` is SB3's `Monitor` plus two methods, `get_episode_state()` and
`set_episode_state()`, which cover the whole worker: the environment's episode
(`NanoEnv.get_episode_state`) AND the monitor's own running totals (rewards of the
episode so far, so that the reward/length reported when the episode ends are the
right ones). They are what lets a resumed training go on in the middle of the
episodes that were running when the checkpoint was written
(see nanogoal_rl/checkpoints.py).

SubprocVecEnv reaches them with `vec_env.env_method("get_episode_state")`: the
call is resolved on the outermost wrapper that has the method, i.e. this one.
"""
import time

from stable_baselines3.common.monitor import Monitor


class ResumableMonitor(Monitor):
    def get_episode_state(self):
        if self.needs_reset:
            raise RuntimeError("No episode in progress (the episode has ended and the env was not reset).")
        return {
            "env": self.env.unwrapped.get_episode_state(),
            "monitor": {
                "rewards": list(self.rewards),
                "total_steps": int(self.total_steps),
                "elapsed": time.time() - self.t_start,
            },
        }

    def set_episode_state(self, state):
        """Restores the episode; returns the observation the agent would see now."""
        observation = self.env.unwrapped.set_episode_state(state["env"])
        monitor = state["monitor"]
        self.rewards = [float(r) for r in monitor["rewards"]]
        self.total_steps = int(monitor["total_steps"])
        self.t_start = time.time() - float(monitor["elapsed"])
        self.needs_reset = False
        return observation
