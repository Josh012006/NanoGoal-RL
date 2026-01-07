import env
import numpy as np
import gymnasium as gym

myEnv = gym.make("Nano-v0", render_mode="human")

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=20)

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    action = np.array([np.random.uniform(2.0, 6.0), np.random.uniform(-np.pi, np.pi)], dtype=np.float32)  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = myEnv.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
myEnv.close()