# This code helps see the performance of the different trained models visually.
# It takes two named arguments: --model is the difficulty for which the model
# was trained, and --seed is the difficulty of the world seed used for the episode.
import numpy as np
import argparse
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
    "--seed", required=True, choices=DIFFICULTIES,
    help="Difficulty of the world seed to use for this episode."
)
args = parser.parse_args()

models = {
    "easy":   "ppo_nanogoal_easy",
    "medium": "ppo_nanogoal_medium",
    "hard":   "ppo_nanogoal_hard",
}
seeds = {
    "easy":   6568, #1296,
    "medium": 1520,
    "hard":   2544,
}

myEnv = env.NanoEnv(render_mode="human")
model = PPO.load("models/" + models[args.model], env=myEnv)

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=seeds[args.seed])

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

    print(f"Episode finished! Total reward: {total_reward}")

except KeyboardInterrupt:
    print("Evaluation stopped by user.")

finally:
    myEnv.close()