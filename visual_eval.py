# This code helps see the performance of the different trained models visually.
# It takes two named arguments: --model is the difficulty for which the model
# was trained, and --seed is the difficulty of the world seed used for the episode.
# Optionally, --seed_value lets you inspect one exact seed number (e.g. a
# specific seed pulled from a results CSV) instead of a difficulty default.
#
# Every episode is ALSO saved as an animated GIF, regardless of whether a real
# display is available. On a machine with a screen, you get both a live
# window AND a saved GIF. On a headless server (e.g. over SSH with no X11),
# set SDL_VIDEODRIVER=dummy first -- no window will actually be visible, but
# the episode is still captured and saved correctly, since frame capture reads
# from pygame's off-screen drawing surface, not from the physical display.
import numpy as np
import argparse
import torch

# Force single-threaded, bit-reproducible PyTorch computation — see eval.py
# for the full rationale. Must be set before any PPO model is loaded.
torch.set_num_threads(1)

import env
from stable_baselines3 import PPO
from PIL import Image
from pathlib import Path

DIFFICULTIES = ["easy", "medium", "hard"]

parser = argparse.ArgumentParser(
    description="Launch a single episode with visual rendering using a trained model, "
                "and save it as a GIF."
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
parser.add_argument(
    "--output", required=False, default=None,
    help="Output GIF path. Defaults to visual_eval_<model>_<seed>.gif"
)
parser.add_argument(
    "--fps", required=False, type=int, default=20,
    help="Playback speed of the saved GIF (frames per second). Default 20."
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
VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

output_path = args.output or str(VIDEOS_DIR / f"visual_eval_{args.model}_{chosen_seed}.gif")

myEnv = env.NanoEnv(render_mode="human")
model = PPO.load("models/" + models[args.model], env=myEnv, device="cpu")  # avoid CPU/GPU non-determinism

# Reset environment to start a new episode
observation, info = myEnv.reset(seed=chosen_seed)
frames = [myEnv._last_frame]

print(f"Using seed: {chosen_seed}")
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

def save_gif():
    # Factored out so it runs from the `finally` block too: an interrupted
    # episode (Ctrl+C, closing the window) still gets saved with whatever
    # frames were captured up to that point, instead of losing the run.
    if len(frames) < 2:
        print("Not enough frames captured, skipping GIF save.")
        return
    duration_ms = int(1000 / args.fps)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        output_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved {len(frames)} frames to: {output_path}")

try:
    while not episode_over:
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = myEnv.step(action)
        frames.append(myEnv._last_frame)

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
    save_gif()