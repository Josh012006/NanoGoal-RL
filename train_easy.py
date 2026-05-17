import env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from checkpoint_callback import KeepLastTwoCheckpoints


myEnv = env.NanoEnv(difficulty="easy")

check_env(myEnv)

myEnv.reset()

checkpoint_callback = KeepLastTwoCheckpoints(
    save_freq=1_000_000,
    save_path="./checkpoints/easy/",
    name_prefix="ppo_easy"
)

# Define and Train the agent
model = PPO(
    "MultiInputPolicy", 
    env=myEnv,
    verbose=1,
    tensorboard_log="./logs/",
    n_steps=10_000
)
model.learn(
    total_timesteps=30_000_000,
    tb_log_name="easy",
    callback=checkpoint_callback
)

# Save the trained agent
model.save("models/ppo_nanogoal_easy")
