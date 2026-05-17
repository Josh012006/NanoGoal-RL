import os
import env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints


# Check the environment first
check_env(env.NanoEnv(difficulty="easy"))

# Automatically detect the number of available CPUs
n_envs = min(os.cpu_count(), 8)
n_steps = 10_000 // n_envs  # keep total steps per iteration constant
print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")


vec_env = make_vec_env(
    lambda: env.NanoEnv(difficulty="easy"),
    n_envs=n_envs
)


checkpoint_callback = KeepLastTwoCheckpoints(
    save_freq=1_000_000,
    save_path="./checkpoints/easy/",
    name_prefix="ppo_easy"
)


# Define and train the agent
model = PPO(
    "MultiInputPolicy", 
    env=vec_env,
    verbose=1,
    tensorboard_log="./logs/",
    n_steps=n_steps
)

model.learn(
    total_timesteps=30_000_000,
    tb_log_name="easy",
    callback=checkpoint_callback
)

# Save the trained agent
model.save("models/ppo_nanogoal_easy")
