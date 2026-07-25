# A program to test the models and evaluate the results

import argparse
import torch

# Force single-threaded, bit-reproducible PyTorch computation. Multi-threaded
# BLAS/OpenMP reductions (matrix multiplies inside the policy network) are NOT
# guaranteed bit-identical run to run, even on the same machine, because the
# order partial sums from different threads get combined depends on OS thread
# scheduling. In this environment, tiny floating-point differences can flip a
# discrete decision (e.g. which side of a wall-collision branch is taken),
# which then compounds over the episode into a completely different rollout.
# This must be set before any PPO model is loaded.
torch.set_num_threads(1)

import json
import env
from stable_baselines3 import PPO
import csv
from pathlib import Path
import numpy as np

DIFFICULTIES = ["easy", "medium", "hard"]
SEED_MODES   = ["easy", "medium", "hard", "mix"]

parser = argparse.ArgumentParser(
    description="Evaluate a trained PPO model over 100 test episodes and save results as a CSV."
)
parser.add_argument(
    "--model", required=True, choices=DIFFICULTIES,
    help="Difficulty the model was trained for."
)
parser.add_argument(
    "--seed", required=True, choices=SEED_MODES,
    help="Difficulty of the world seeds to test on ('mix' combines all three categories)."
)
args = parser.parse_args()

model_difficulty = args.model
seed_mode        = args.seed


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

def _build_test_set(all_category, train_set, n=500):
    candidates = np.array([s for s in all_category if s not in train_set])
    k          = min(n, len(candidates))
    idx        = _test_rng.choice(len(candidates), size=k, replace=False)
    return candidates[idx].tolist()

test_easy_seeds   = _build_test_set(_all_seeds["easy"],   train_easy)
test_medium_seeds = _build_test_set(_all_seeds["medium"], train_medium)
test_hard_seeds   = _build_test_set(_all_seeds["hard"],   train_hard)

test_sets = {
    "easy":   test_easy_seeds,
    "medium": test_medium_seeds,
    "hard":   test_hard_seeds,
    "mix":    test_easy_seeds + test_medium_seeds + test_hard_seeds,
}

# Fixed seed so the TESTING ORDER is also reproducible run to run, not just
# the set of seeds tested. Previously this used the unseeded global np.random
# module, meaning two separate eval.py runs would test the same 500 seeds but
# in a different order each time -- harmless for aggregate stats, but it made
# comparing two runs row-by-row (e.g. to verify reproducibility) misleading,
# since "episode 0" wasn't the same seed between runs.
_shuffle_rng = np.random.default_rng(24680)
test_set = _shuffle_rng.permutation(test_sets[seed_mode])

myEnv = env.NanoEnv()
model = PPO.load("models/ppo_nanogoal_" + model_difficulty, env=myEnv, device="cpu")  # avoid CPU/GPU non-determinism


folder = model_difficulty
Path("results/" + folder).mkdir(parents=True, exist_ok=True)
with open("results/" + folder + "/ppo_eval_" + seed_mode + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "seed", "return", "length", "success", "terminated", "truncated", "init_dist_goal", "best_dist_goal", "final_dist_goal"])

    for episode in range(500):
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