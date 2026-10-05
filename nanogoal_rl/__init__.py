"""NanoGoal-RL: simulated 2D nanorobot navigating a vessel-like environment."""
from gymnasium.envs.registration import register

from .env import NanoEnv

register(id="Nano-v0", entry_point="nanogoal_rl.env:NanoEnv")

__all__ = ["NanoEnv"]