# This code helps see the performance of the different trained models visually.
# It takes two named arguments: --model is the difficulty for which the model
# was trained, and --seed is the difficulty of the world seed used for the episode.
# Optionally, --seed_value lets you inspect one exact seed number (e.g. a
# specific seed pulled from a results CSV) instead of a difficulty default.
import numpy as np
import argparse
import torch

# Force single-threaded, bit-reproducible PyTorch computation — see eval.py
# for the full rationale. Must be set before any PPO model is loaded.
torch.set_num_threads(1)

import env
from stable_baselines3 import PPO

DIFFICULTIES = ["easy", "medium", "hard"]

parser = argparse.ArgumentParser(
    description="Launch a single episode with visual rendering using a trained model."
)
parser.add_argument(
    "--model", required=True, choices=DIFFICULTIES,
    help="Difficulty the model was trained for."
)
parser.add_argument(
    "--seed", required=False, choices=DIFFICULTIES, default=None,
    help="Difficulty of the world seed to use (a fixed default seed for that difficulty)."
)
parser.add_argument(
    "--seed_value", required=False, type=int, default=None,
    help="Exact seed number to use instead of a difficulty default — "
         "e.g. to inspect a specific seed pulled from a results CSV."
)
args = parser.parse_args()

if args.seed is None and args.seed_value is None:
    parser.error("Provide either --seed <easy|medium|hard> or --seed_value <int>.")

models = {
    "easy":   "ppo_nanogoal_easy",
    "medium": "ppo_nanogoal_medium",
    "hard":   "ppo_nanogoal_hard",
}
default_seeds = {
    "easy":   1520,
    "medium": 6568,
    "hard":   1296,
}

chosen_seed = args.seed_value if args.seed_value is not None else default_seeds[args.seed]

myEnv = env.NanoEnv(render_mode="human")
model = PPO.load("models/" + models[args.model], env=myEnv, device="cpu")  # avoid CPU/GPU non-determinism

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=chosen_seed)

print(f"Using seed: {chosen_seed}")
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

try:
    while not episode_over:
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = myEnv.step(action)

        total_reward += reward
        episode_over = terminated or truncated
        print(
            "orientation:",
            np.degrees(myEnv._orientation),
            "direction:",
            myEnv._get_obs()["mvt"][1:]
        )

    print(f"Episode finished! Total reward: {total_reward}, success: {info['is_success']}")

except KeyboardInterrupt:
    print("Evaluation stopped by user.")

finally:
    myEnv.close()