import env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


myEnv = env.NanoEnv()

check_env(myEnv)

myEnv.reset()

# Define and Train the agent
model = PPO(
    "MultiInputPolicy", 
    myEnv,
    verbose=1,
    tensorboard_log="./logs/"
)
model.learn(total_timesteps=800_000)

# Save the trained agent
model.save("models/ppo_nanogoal")
