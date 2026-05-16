# precompute_cache.py
# Run this script once after classify_seeds.py:
#   python precompute_cache.py
# It generates topology_cache (shelve files) containing for each seed:
#   - the topology (np.ndarray of shape (125, 125))
#   - the filtered available space (np.ndarray of shape (N, 2))

import json
import shelve
import numpy as np
import env as E
from utils import main_related_component


def precompute(seed: int, environment: E.NanoEnv):
    """
    Reproduce exactly the world generation phase from reset(),
    without placing the agent or the target — just the topology and free space.

    new_seed starts at seed+1 and increments until it finds
    a topology with enough available space (> 100 cells).
    """
    new_seed = 1 + int(seed)
    while True:
        topology  = environment._generate_logical_topology(new_seed)
        available = main_related_component(topology)
        available = environment._filter_by_clearance(
            available,
            max(environment._agent_radius, environment._target_radius)
        )
        new_seed += 1
        if len(available) > 100:
            # Stack the list of np.array([i, j]) into a single (N, 2) array
            # instead of storing thousands of individual np.arrays — drastically
            # reduces serialization overhead and final file size
            available_arr = np.stack(available, axis=0).astype(np.float32)
            return topology, available_arr


if __name__ == "__main__":
    with open("seeds.json") as f:
        all_seeds = json.load(f)

    all_training_seeds = (
        all_seeds["easy"] + all_seeds["medium"] + all_seeds["hard"]
    )

    environment = E.NanoEnv()

    print(f"Precomputing {len(all_training_seeds)} seeds...")

    # shelve writes each entry directly to disk — no MemoryError on large datasets
    with shelve.open("topology_cache") as cache:
        for i, seed in enumerate(all_training_seeds):
            topology, available_arr = precompute(seed, environment)

            # shelve requires string keys
            cache[str(seed)] = {
                "topology":  topology,       # np.ndarray (125, 125)
                "available": available_arr,  # np.ndarray (N, 2) — compact
            }

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(all_training_seeds)}")

    environment.close()
    print(f"topology_cache generated ({len(all_training_seeds)} entries).")