import env
from stable_baselines3 import PPO

import csv

myEnv = env.NanoEnv()
model = PPO.load("models/ppo_nanogoal", env=myEnv)

with open("results/ppo_eval.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "return", "length", "success", "terminated", "truncated", "min_dist_goal", "final_dist_goal"])

    for episode in range(20):
        obs, info = myEnv.reset()

        total_reward = 0
        step = 0
        min_dist = info["distance"]

        # Run the episode
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = myEnv.step(action)

            total_reward += reward
            step += 1
            min_dist = min(min_dist, info["distance"])


        success = info["is_success"]
        final_dist = info["distance"]
        
        # Save the results
        writer.writerow([episode, total_reward, step, success, terminated, truncated, min_dist, final_dist])