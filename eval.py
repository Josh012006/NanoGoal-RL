# A program to test the models and evaluate the results

import json
import env
from stable_baselines3 import PPO
import csv
from pathlib import Path
import numpy as np
import sys


with open("seeds.json") as f:
    _all_seeds = json.load(f)

# ── Rebuild exactly the same training split as in env.py ──────
_loader_rng = np.random.default_rng(99999)  # same seed as in env.py

def _sample_category(seeds_list, pct=0.40):
    arr = np.array(seeds_list)
    k   = max(1, int(len(arr) * pct))
    idx = _loader_rng.choice(len(arr), size=k, replace=False)
    return set(arr[idx].tolist())

train_easy   = _sample_category(_all_seeds["easy"])
train_medium = _sample_category(_all_seeds["medium"])
train_hard   = _sample_category(_all_seeds["hard"])

# ── Test sets = seeds classified NOT used during training ──────────────
_test_rng = np.random.default_rng(77777)  

def _build_test_set(all_category, train_set, n=100):
    candidates = np.array([s for s in all_category if s not in train_set])
    k          = min(n, len(candidates))
    idx        = _test_rng.choice(len(candidates), size=k, replace=False)
    return candidates[idx].tolist()

test_easy_seeds   = _build_test_set(_all_seeds["easy"],   train_easy)
test_medium_seeds = _build_test_set(_all_seeds["medium"], train_medium)
test_hard_seeds   = _build_test_set(_all_seeds["hard"],   train_hard)

model_difficulty = int(sys.argv[1]) # 0 for easy, 1 for medium and 2 for hard
difficulty_mode = int(sys.argv[2]) # 0 for easy, 1 for medium, 2 for hard and 3 for mix


rng = np.random
test_set = rng.permutation(test_easy_seeds if difficulty_mode == 0 else test_medium_seeds if difficulty_mode == 1 else test_hard_seeds if difficulty_mode == 2 else (test_easy_seeds + test_medium_seeds + test_hard_seeds))
difficulty = "easy" if model_difficulty == 0 else "medium" if model_difficulty == 1 else "hard"

myEnv = env.NanoEnv()
model = PPO.load("models/ppo_nanogoal_" + difficulty, env=myEnv)


folder = difficulty
Path("results/" + folder).mkdir(parents=True, exist_ok=True)
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