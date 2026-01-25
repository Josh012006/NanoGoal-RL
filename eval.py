# A program to test the models and evaluate the results

import env
from stable_baselines3 import PPO
import csv
from pathlib import Path
import numpy as np
import sys

test_easy_seeds = [0, 1, 2, 3, 5, 7, 8, 10, 11, 13, 14, 16, 18, 20, 21, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                   64, 65, 67, 70, 72, 73, 75, 76, 77, 78, 79, 81, 83, 84, 89, 91, 93, 94, 95, 97, 98, 100, ]
test_medium_seeds = [4, 12, 19, 22, 25, 35, 43, 44, 63, 68, 69, 71, 80, 82, 85, 87, 88, 92, ]
test_hard_seeds = [6, 9, 15, 17, 24, 41, 49, 66, 74, 86, 90, 96, 99, ]

entry = int(sys.argv[1]) # 0 for easy, 1 for medium, 2 for hard and 3 for mix
rng = np.random
test_set = rng.permutation(test_easy_seeds if entry == 0 else test_medium_seeds if entry == 1 else test_hard_seeds if entry == 2 else test_easy_seeds + test_medium_seeds + test_hard_seeds)
difficulty = "easy" if entry == 0 else "medium" if entry == 1 else "hard"

myEnv = env.NanoEnv()
model = PPO.load("models/ppo_nanogoal_" + difficulty, env=myEnv)


Path("results").mkdir(parents=True, exist_ok=True)

with open("results/ppo_eval.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "seed", "return", "length", "success", "terminated", "truncated", "init_dist_goal", "best_dist_goal", "final_dist_goal"])

    for episode in range(500):
        seed = rng.choice(test_set)
        obs, info = myEnv.reset(seed=seed)
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
        writer.writerow([episode, seed, total_reward, step, success, terminated, truncated, init_dist, best_dist, final_dist])