# classify_seeds.py
import json
import heapq
import numpy as np
import env

SEED_RANGE = range(10000)

TURN_THRESHOLD_DEG = 45   # minimum angle change to count as a real wall detour
ALPHA              = 30.0 # penalty per turn in the composite score (used for intra-category sorting only)


# ── A* returning path + turn count ────────────────────────────────────────────

def heuristic(a, b):
    """Manhattan distance — admissible heuristic for A* on a grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path_and_turns(grid, start, goal, agent_radius,
                          turn_threshold_deg=TURN_THRESHOLD_DEG):
    """
    Runs A* from start to goal on the discrete grid.
    Returns (path_length, nb_turns) where:
      - path_length : total cost of the optimal path (np.inf if unreachable)
      - nb_turns    : number of significant direction changes (≥ turn_threshold_deg)

    A 'turn' is counted when the angle between two consecutive movement
    vectors exceeds the threshold — this captures genuine wall detours,
    not the small diagonal adjustments inherent to grid movement.
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
        return np.inf, 0

    # Min-heap: (f, g, node)
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

    g_scores  = {start: 0.0}
    came_from = {}  # to reconstruct the path

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

            # ── Count significant direction changes ───────────────────────────
            threshold_rad = np.radians(turn_threshold_deg)
            nb_turns = 0
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
                    nb_turns += 1
                    # Skip a few nodes after a turn to avoid counting the same
                    # detour multiple times (the path wobbles around a corner)
                    i += 3
                else:
                    i += 1

            return g, nb_turns

        if g > g_scores.get(current, np.inf):
            continue

        for (di, dj), cost in neighbors:
            ni, nj    = current[0] + di, current[1] + dj
            neighbor  = (ni, nj)
            if not is_free(ni, nj):
                continue
            new_g = g + cost
            if new_g < g_scores.get(neighbor, np.inf):
                g_scores[neighbor]  = new_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (new_g + heuristic(neighbor, goal), new_g, neighbor))

    return np.inf, 0  # unreachable


# ── Difficulty score ───────────────────────────────────────────────────────────

def score(seed: int, environment: env.NanoEnv):
    """
    Returns (path_length, nb_turns, composite_score) for a given seed.
    composite_score = path_length + ALPHA * nb_turns
    Used for intra-category sorting only — classification is based on nb_turns.
    """
    environment.reset(seed=seed)
    start = (int(environment._agent_location[0]), int(environment._agent_location[1]))
    goal  = (int(environment._target_location[0]), int(environment._target_location[1]))

    path_length, nb_turns = astar_path_and_turns(
        environment._vessel_topology, start, goal, environment._agent_radius
    )

    if path_length == np.inf:
        return np.inf, 0, np.inf

    composite = path_length + ALPHA * nb_turns
    return path_length, nb_turns, composite


# ── Classification and export ─────────────────────────────────────────────────

if __name__ == "__main__":
    environment = env.NanoEnv()

    print(f"Computing scores for {len(SEED_RANGE)} seeds...")
    results     = {}  # seed → (path_length, nb_turns, composite)
    unreachable = 0

    for s in SEED_RANGE:
        pl, nt, cs = score(s, environment)
        if pl == np.inf:
            unreachable += 1
        else:
            results[s] = (pl, nt, cs)

        if (s + 1) % 500 == 0:
            print(f"  {s + 1}/{len(SEED_RANGE)}")

    environment.close()
    print(f"{len(results)} reachable seeds, {unreachable} ignored (unreachable).")

    # ── Explicit classification by number of turns ────────────────────────────
    # Easy   : 0 turns   — near straight line to goal
    # Medium : 1–2 turns — requires learning to navigate around walls
    # Hard   : 3+ turns  — requires combining multiple navigation skills
    easy_seeds   = {s: v for s, v in results.items() if v[1] == 0}
    medium_seeds = {s: v for s, v in results.items() if 1 <= v[1] <= 2}
    hard_seeds   = {s: v for s, v in results.items() if v[1] >= 3}

    # Sort intra-category by composite score (shortest/simplest first)
    easy_sorted   = sorted(easy_seeds.keys(),   key=lambda s: easy_seeds[s][2])
    medium_sorted = sorted(medium_seeds.keys(), key=lambda s: medium_seeds[s][2])
    hard_sorted   = sorted(hard_seeds.keys(),   key=lambda s: hard_seeds[s][2])

    seeds = {
        "easy":   easy_sorted,
        "medium": medium_sorted,
        "hard":   hard_sorted,
    }

    with open("seeds.json", "w") as f:
        json.dump(seeds, f, indent=2)

    print(f"\nseeds.json generated:")
    print(f"  easy   (0 turns) : {len(easy_sorted):>5} seeds")
    print(f"  medium (1–2 turns): {len(medium_sorted):>5} seeds")
    print(f"  hard   (3+ turns) : {len(hard_sorted):>5} seeds")

    # Distribution overview
    turn_counts = [v[1] for v in results.values()]
    print(f"\nTurn distribution:")
    for t in range(8):
        n = sum(1 for tc in turn_counts if tc == t)
        print(f"  {t} turns: {n} seeds")
    n_more = sum(1 for tc in turn_counts if tc >= 8)
    if n_more:
        print(f"  8+ turns: {n_more} seeds")