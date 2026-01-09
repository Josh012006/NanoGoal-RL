import env
from stable_baselines3 import PPO
import csv
from pathlib import Path



myEnv = env.NanoEnv()
model = PPO.load("models/ppo_nanogoal", env=myEnv)


Path("results").mkdir(parents=True, exist_ok=True)

with open("results/ppo_eval.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "return", "length", "success", "terminated", "truncated", "init_dist_goal", "best_dist_goal", "final_dist_goal"])

    for episode in range(300):
        obs, info = myEnv.reset()
        terminated = False
        truncated = False

        total_reward = 0
        step = 0
        init_dist = info["distance"]

        # Run the episode
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = myEnv.step(action)

            total_reward += reward
            step += 1


        success = info["is_success"]
        best_dist = info["best_dist"]
        final_dist = info["distance"]
        
        # Save the results
        writer.writerow([episode, total_reward, step, success, terminated, truncated, init_dist, best_dist, final_dist])