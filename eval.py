# A program to test the models and evaluate the results

import env
from stable_baselines3 import PPO
import csv
from pathlib import Path
import numpy as np
import sys

test_easy_seeds = [0, 1, 2, 3, 5, 7, 8, 10, 11, 13, 14, 16, 18, 20, 21, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                   64, 65, 67, 70, 72, 73, 75, 76, 77, 78, 79, 81, 83, 84, 89, 91, 93, 94, 95, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 113, 114, 115, 116, 117, 195, 265, 721, 726, 729, 
                   1001, 2002, 2006, 2011, 2012, 2020, 2029, 2565, 7011, 8188, 9151] 

test_medium_seeds = [4, 12, 19, 22, 25, 35, 43, 44, 63, 68, 69, 71, 80, 82, 85, 87, 88, 92, 119, 121, 134, 135, 138, 139, 142, 143, 144, 146, 148, 149, 151, 152, 154, 167, 169, 171, 179, 182, 190, 192, 
                     198, 202, 204, 206, 209, 212, 220, 222, 225, 226, 232, 233, 238, 241, 245, 246, 252, 255, 257, 262, 263, 269, 270, 274, 276, 277, 282, 285, 287, 290, 293, 298, 299, 301, 304, 307, 
                     310, 318, 319, 322, 326, 327, 328, 331, 336, 337, 338, 339, 340, 343, 347, 665, 728, 989, 1004, 1005, 2003, 2022, 2023, 2222] 

test_hard_seeds = [6, 9, 15, 17, 24, 41, 49, 66, 74, 86, 90, 96, 99, 108, 109, 111, 132, 133, 136, 147, 155, 165, 173, 178, 180, 185, 186, 187, 193, 200, 228, 231, 235, 242, 248, 251, 253, 261, 264, 271,
                   275, 280, 281, 288, 289, 306, 312, 315, 332, 341, 342, 344, 345, 349, 350, 367, 372, 374, 382, 384, 385, 386, 394, 406, 407, 418, 424, 432, 433, 436, 448, 455, 456, 496, 506, 508, 522, 
                   530, 538, 546, 563, 564, 580, 584, 625, 641, 642, 643, 652, 661, 663, 671, 676, 705, 707, 709, 710, 711, 717, 718]



model_difficulty = int(sys.argv[1]) # 0 for easy, 1 for medium and 2 for hard
difficulty_mode = int(sys.argv[2]) # 0 for easy, 1 for medium, 2 for hard and 3 for mix


rng = np.random
test_set = rng.permutation(test_easy_seeds if difficulty_mode == 0 else test_medium_seeds if difficulty_mode == 1 else test_hard_seeds if difficulty_mode == 2 else (test_easy_seeds + test_medium_seeds + test_hard_seeds))
difficulty = "easy" if model_difficulty == 0 else "medium" if model_difficulty == 1 else "hard"

myEnv = env.NanoEnv()
model = PPO.load("models/ppo_nanogoal_" + difficulty, env=myEnv)


Path("results").mkdir(parents=True, exist_ok=True)

folder = difficulty
termination = "easy" if difficulty_mode == 0 else "medium" if difficulty_mode == 1 else "hard" if difficulty_mode == 2 else "mix"
with open("results/" + folder + "/ppo_eval_" + termination + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "seed", "return", "length", "success", "terminated", "truncated", "init_dist_goal", "best_dist_goal", "final_dist_goal"])

    for episode in range(100):
        seed = test_set[episode]
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