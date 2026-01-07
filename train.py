import env
import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_checker import check_env


myEnv = env.NanoEnv()

check_env(myEnv)
# myEnv = gym.make("Nano-v0", render_mode="human")

# # Reset environment to start a new episode
# observation, info = myEnv.reset(seed=100)

# print(f"Starting observation: {observation}")

# episode_over = False
# total_reward = 0

# while not episode_over:
#     action = myEnv.action_space.sample()  # Random action for now - real agents will be smarter!

#     # Take the action and see what happens
#     observation, reward, terminated, truncated, info = myEnv.step(action)

#     total_reward += reward
#     episode_over = terminated or truncated

# print(f"Episode finished! Total reward: {total_reward}")
# myEnv.close()