import env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


myEnv = env.NanoEnv(difficulty="easy")

check_env(myEnv)

myEnv.reset()

# Define and Train the agent
model = PPO(
    "MultiInputPolicy", 
    env=myEnv,
    verbose=1,
    tensorboard_log="./logs/",
    n_steps=10_000
)
model.learn(
    total_timesteps=12_000_000, # approximatively 20000 episodes
    tb_log_name="easy"
)

# Save the trained agent
model.save("models/ppo_nanogoal_easy")
