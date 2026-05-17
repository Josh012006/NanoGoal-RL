import os
import env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints

n_envs = min(os.cpu_count(), 8)
n_steps = 10_000 // n_envs
print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

vec_env = make_vec_env(
    lambda: env.NanoEnv(difficulty="medium"),
    n_envs=n_envs
)

checkpoint_callback = KeepLastTwoCheckpoints(
    save_freq=1_000_000,
    save_path="./checkpoints/medium/",
    name_prefix="ppo_medium"
)

model = PPO.load(
    "models/ppo_nanogoal_easy",
    env=vec_env,
    custom_objects={"n_steps": n_steps}
)

model.learn(
    total_timesteps=100_000_000, 
    reset_num_timesteps=False,
    tb_log_name="medium",
    callback=checkpoint_callback
)

model.save("models/ppo_nanogoal_medium")
