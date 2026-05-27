import os
import env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints


def make_env():
    return env.NanoEnv(difficulty="hard")


if __name__ == "__main__":
    # Automatically detect the number of available CPUs
    n_envs = min(os.cpu_count(), 8)
    n_steps = 20_000 // n_envs  # increased to reduce backprop proportion vs rollout
    print(f"Running with {n_envs} parallel environments ({n_steps} steps each)")

    # SubprocVecEnv spawns one process per env, enabling true CPU parallelism
    vec_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )

    checkpoint_callback = KeepLastTwoCheckpoints(
        save_freq=1_000_000,
        save_path="./checkpoints/hard/",
        name_prefix="ppo_hard"
    )

    model = PPO.load(
        "models/ppo_nanogoal_medium",
        env=vec_env,
        custom_objects={"n_steps": n_steps, "learning_rate": 5e-5}
    )

    model.learn(
        total_timesteps=280_000_000,
        reset_num_timesteps=False,
        tb_log_name="hard",
        callback=checkpoint_callback
    )

    model.save("models/ppo_nanogoal_hard")