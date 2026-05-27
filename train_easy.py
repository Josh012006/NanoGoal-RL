import os
import env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints


# Check the environment first
check_env(env.NanoEnv(difficulty="easy"))

# Automatically detect the number of available CPUs
n_envs = min(os.cpu_count(), 8)
n_steps = 20_000 // n_envs  # increased to reduce backprop proportion vs rollout
print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

def make_env():
    return env.NanoEnv(difficulty="easy")

# SubprocVecEnv spawns one process per env, enabling true CPU parallelism
vec_env = make_vec_env(
    make_env,
    n_envs=n_envs,
    vec_env_cls=SubprocVecEnv
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
    total_timesteps=50_000_000,
    tb_log_name="easy",
    callback=checkpoint_callback
)

# Save the trained agent
model.save("models/ppo_nanogoal_easy")