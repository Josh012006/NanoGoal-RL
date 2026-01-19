import env
from stable_baselines3 import PPO

env_medium = env.NanoEnv(difficulty="medium")


model = PPO.load(
    "models/ppo_nanogoal_easy",
    env=env_medium
)

model.learn(
    total_timesteps=450_000,
    reset_num_timesteps=False,
    tb_log_name="medium"
)

model.save("models/ppo_nanogoal_medium")
