# A file just to test and observe more freely with different seeds the environment and 
# the effect of the actions.

import env
import numpy as np

import sys


myEnv = env.NanoEnv(render_mode="human", max_v=80.0)

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=int(sys.argv[1]))

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

i = 0
while not episode_over:
    test_action = np.array([0.0, 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = myEnv.step(test_action)

    total_reward += reward
    episode_over = terminated or truncated
    i += 1

print(f"Episode finished! Total reward: {total_reward}")
myEnv.close()