import env
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints

env_medium = env.NanoEnv(difficulty="medium")

checkpoint_callback = KeepLastTwoCheckpoints(
    save_freq=1_000_000,
    save_path="./checkpoints/medium/",
    name_prefix="ppo_medium"
)

model = PPO.load(
    "models/ppo_nanogoal_easy",
    env=env_medium
)

model.learn(
    total_timesteps=100_000_000, 
    tb_log_name="medium",
    callback=checkpoint_callback
)

model.save("models/ppo_nanogoal_medium")
