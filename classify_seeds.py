# classify_seeds.py
import json
import heapq
import numpy as np
import env

SEED_RANGE = range(10000)

TURN_THRESHOLD_DEG = 45    # minimum angle to count as a real direction change
EASY_MAX_DEG       = 46    # easy   : total deviation < 46°  (0 real detours, only diagonal adjustments)
MEDIUM_MAX_DEG     = 270   # medium : total deviation 46°–270° (1–2 real detours)
                           # hard   : total deviation > 270° (complex navigation)


# ── A* returning path + total angular deviation ────────────────────────────────

def heuristic(a, b):
    """Manhattan distance — admissible heuristic for A* on a grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path_and_deviation(grid, start, goal, agent_radius,
                              turn_threshold_deg=TURN_THRESHOLD_DEG):
    """
    Runs A* from start to goal on the discrete grid.
    Returns (path_length, total_deviation_deg) where:
      - path_length         : total A* cost (np.inf if unreachable)
      - total_deviation_deg : sum of all significant direction changes in degrees.
                              Small wiggles (< turn_threshold_deg) are ignored.
                              This captures the true navigational complexity —
                              a 170° U-turn counts far more than a 46° nudge.
    """
    size  = len(grid)
    int_r = int(np.ceil(agent_radius))

    def is_free(i, j):
        if not (0 <= i < size and 0 <= j < size):
            return False
        if grid[i][j] == 1:
            return False
        for di in range(-int_r, int_r + 1):
            for dj in range(-int_r, int_r + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size and grid[ni][nj] == 1:
                    return False
        return True

    if not is_free(*start) or not is_free(*goal):
        return np.inf, 0.0

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    g_scores  = {start: 0.0}
    came_from = {}

    neighbors = [
        ((-1,  0), 1.0),   (( 1,  0), 1.0),
        (( 0, -1), 1.0),   (( 0,  1), 1.0),
        ((-1, -1), 1.414), ((-1,  1), 1.414),
        (( 1, -1), 1.414), (( 1,  1), 1.414),
    ]

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current == goal:
            # ── Reconstruct path ──────────────────────────────────────────────
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()

            # ── Compute total angular deviation ───────────────────────────────
            # We accumulate only angles above the threshold — small grid
            # wobbles (diagonal steps, minor adjustments) are ignored.
            threshold_rad   = np.radians(turn_threshold_deg)
            total_deviation = 0.0
            i = 1
            while i < len(path) - 1:
                v1 = np.array(path[i])     - np.array(path[i - 1])
                v2 = np.array(path[i + 1]) - np.array(path[i])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    i += 1
                    continue
                cos_a = np.dot(v1, v2) / (n1 * n2)
                angle = np.arccos(np.clip(cos_a, -1.0, 1.0))
                if angle >= threshold_rad:
                    total_deviation += np.degrees(angle)
                    i += 3  # skip a few nodes to avoid double-counting a corner
                else:
                    i += 1

            return g, total_deviation

        if g > g_scores.get(current, np.inf):
            continue

        for (di, dj), cost in neighbors:
            ni, nj   = current[0] + di, current[1] + dj
            neighbor = (ni, nj)
            if not is_free(ni, nj):
                continue
            new_g = g + cost
            if new_g < g_scores.get(neighbor, np.inf):
                g_scores[neighbor]  = new_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (new_g + heuristic(neighbor, goal), new_g, neighbor))

    return np.inf, 0.0


# ── Score per seed ─────────────────────────────────────────────────────────────

def score(seed: int, environment: env.NanoEnv):
    """
    Returns (path_length, total_deviation_deg) for a given seed.
    Returns (np.inf, 0) if the goal is unreachable.
    """
    environment.reset(seed=seed)
    start = (int(environment._agent_location[0]), int(environment._agent_location[1]))
    goal  = (int(environment._target_location[0]), int(environment._target_location[1]))

    return astar_path_and_deviation(
        environment._vessel_topology, start, goal, environment._agent_radius
    )


# ── Classification and export ─────────────────────────────────────────────────

if __name__ == "__main__":
    environment = env.NanoEnv()

    print(f"Computing scores for {len(SEED_RANGE)} seeds...")
    results     = {}  # seed → (path_length, total_deviation_deg)
    unreachable = 0

    for s in SEED_RANGE:
        pl, dev = score(s, environment)
        if pl == np.inf:
            unreachable += 1
        else:
            results[s] = (pl, dev)

        if (s + 1) % 500 == 0:
            print(f"  {s + 1}/{len(SEED_RANGE)}")

    environment.close()
    print(f"{len(results)} reachable seeds, {unreachable} ignored (unreachable).")

    # ── Classification by total angular deviation ─────────────────────────────
    # Easy   : < 60°   — near straight line, no real detour needed
    # Medium : 60–200° — 1 to 2 significant detours around walls
    # Hard   : > 200°  — complex navigation, multiple combined skills needed
    easy_seeds   = {s: v for s, v in results.items() if v[1] <  EASY_MAX_DEG}
    medium_seeds = {s: v for s, v in results.items() if EASY_MAX_DEG <= v[1] < MEDIUM_MAX_DEG}
    hard_seeds   = {s: v for s, v in results.items() if v[1] >= MEDIUM_MAX_DEG}

    # Sort intra-category by total deviation (simplest first)
    easy_sorted   = sorted(easy_seeds.keys(),   key=lambda s: easy_seeds[s][1])
    medium_sorted = sorted(medium_seeds.keys(), key=lambda s: medium_seeds[s][1])
    hard_sorted   = sorted(hard_seeds.keys(),   key=lambda s: hard_seeds[s][1])

    seeds = {
        "easy":   easy_sorted,
        "medium": medium_sorted,
        "hard":   hard_sorted,
    }

    with open("seeds.json", "w") as f:
        json.dump(seeds, f, indent=2)

    print(f"\nseeds.json generated:")
    print(f"  easy   (< {EASY_MAX_DEG}°)   : {len(easy_sorted):>5} seeds  — straight line, no real detour")
    print(f"  medium ({EASY_MAX_DEG}°–{MEDIUM_MAX_DEG}°): {len(medium_sorted):>5} seeds  — 1–2 wall detours")
    print(f"  hard   (> {MEDIUM_MAX_DEG}°)  : {len(hard_sorted):>5} seeds  — complex navigation")

    # ── Deviation distribution overview ───────────────────────────────────────
    deviations = [v[1] for v in results.values()]
    print(f"\nDeviation distribution (degrees):")
    brackets = [(0, 30), (30, 60), (60, 90), (90, 120),
                (120, 150), (150, 200), (200, 300), (300, 500), (500, 9999)]
    for lo, hi in brackets:
        n = sum(1 for d in deviations if lo <= d < hi)
        label = f"{lo}°–{hi}°" if hi < 9999 else f"{lo}°+"
        print(f"  {label:>12} : {n:>5} seeds")

    deviations_arr = np.array(deviations)
    print(f"\nStats: mean={deviations_arr.mean():.1f}° | "
          f"median={np.median(deviations_arr):.1f}° | "
          f"p25={np.percentile(deviations_arr, 25):.1f}° | "
          f"p75={np.percentile(deviations_arr, 75):.1f}° | "
          f"max={deviations_arr.max():.1f}°")