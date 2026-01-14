import env
import numpy as np
from stable_baselines3 import PPO

myEnv = env.NanoEnv(render_mode="human", max_v=80.0)
model = PPO.load("models/ppo_nanogoal", env=myEnv)

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=30)

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

i = 0
while not episode_over:
    test_action = np.array([0.0375, 0.0 if i == 0 else 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = myEnv.step(test_action)

    total_reward += reward
    episode_over = terminated or truncated
    i += 1

print(f"Episode finished! Total reward: {total_reward}")
myEnv.close()