import env
from stable_baselines3 import PPO

myEnv = env.NanoEnv(render_mode="human")
model = PPO.load("models/ppo_nanogoal", env=myEnv)

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=42)

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    action, _ = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = myEnv.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
myEnv.close()