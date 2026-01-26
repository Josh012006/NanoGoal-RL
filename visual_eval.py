# This code helps see the performance of the different trained models visually. It take
# to arguments. The first one represents the difficulty for which the model was trained and the
# second one is the difficulty of the seed used. We will use 0 for easy, 1 for medium and 2 for hard
# in the two cases. 

import env
import sys
from stable_baselines3 import PPO


model_difficulty = int(sys.argv[1]) 
seed_difficulty = int(sys.argv[2])

models = ["ppo_nanogoal_easy", "ppo_nanogoal_medium", "ppo_nanogoal_hard"]
seeds = [30, 665, 775]

myEnv = env.NanoEnv(render_mode="human")
model = PPO.load("models/" + models[model_difficulty], env=myEnv)

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=seeds[seed_difficulty])

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