import env
from stable_baselines3 import PPO

env_hard = env.NanoEnv(difficulty="hard")

model = PPO.load(
    "models/ppo_nanogoal_medium",
    env=env_hard
)

model.learn(
    total_timesteps=100_000_000,  # approximatively 170000 episodes
    reset_num_timesteps=False,
    tb_log_name="hard"
)

model.save("models/ppo_nanogoal_hard")
