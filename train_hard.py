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

    # Use RUN_ID from environment for unique checkpoint folder per run
    run_id = os.environ.get("RUN_ID", "local")
    checkpoint_path = f"./checkpoints/hard/{run_id}/"
    print(f"Checkpoint path: {checkpoint_path}")

    checkpoint_callback = KeepLastTwoCheckpoints(
        save_freq=1_000_000,
        save_path=checkpoint_path,
        name_prefix="ppo_hard"
    )

    # n_epochs=20 — maximum reuse per rollout for complex multi-detour navigation
    model = PPO.load(
        "models/ppo_nanogoal_medium",
        env=vec_env,
        custom_objects={"n_steps": n_steps, "learning_rate": 5e-5, "n_epochs": 20}
    )

    model.learn(
        total_timesteps=300_000_000,
        reset_num_timesteps=False,
        tb_log_name="hard",
        callback=checkpoint_callback
    )

    model.save("models/ppo_nanogoal_hard")