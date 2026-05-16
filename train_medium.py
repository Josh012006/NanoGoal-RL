import env
from stable_baselines3 import PPO

env_medium = env.NanoEnv(difficulty="medium")


model = PPO.load(
    "models/ppo_nanogoal_easy",
    env=env_medium
)

model.learn(
    total_timesteps=60_000_000, # approximatively 100000 episodes
    tb_log_name="medium"
)

model.save("models/ppo_nanogoal_medium")
