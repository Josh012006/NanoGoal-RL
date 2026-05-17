import env
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints

env_hard = env.NanoEnv(difficulty="hard")

checkpoint_callback = KeepLastTwoCheckpoints(
    save_freq=1_000_000,
    save_path="./checkpoints/hard/",
    name_prefix="ppo_hard"
)

model = PPO.load(
    "models/ppo_nanogoal_medium",
    env=env_hard
)

model.learn(
    total_timesteps=280_000_000,  
    tb_log_name="hard",
    callback=checkpoint_callback
)

model.save("models/ppo_nanogoal_hard")
