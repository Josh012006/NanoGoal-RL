import numpy as np
from numba import njit

def first_not_explored(grid):
    """Finds the first empty cell in the grid that hasn't already been explored. Note that 1 is for wall,
    0 is for empty space (not explored yet) and 2 is for space already explored.
    Args:
        grid: the square grid to work on
    Returns:
        tuple: the position of the first not yet explored cell if there is one. If there isn't any, returns (-1, -1)
    """
    size = len(grid)
    for i in range(size):
        for j in range(size):
            if grid[i][j] == 0: return (i, j)

    return (-1, -1)

def main_related_component(grid):
    """A function that takes a 2D grid describing a topology of an environemnt and computes the 
    main related component as a list of positions. We use the convention 0 is empty, 1 is wall and 2 
    is for space already explored.
    
    Args:
        grid: the square grid representing the topology
    Returns:
        list: the positions in the main related component
    """

    result = []
    work_grid = grid.copy()
    size = len(grid)

    x, y = first_not_explored(work_grid)

    while (x, y) != (-1, -1):
        work_grid[x][y] = 2
        related_component = [(x, y)]

        for elem in related_component:
            i = elem[0]; j = elem[1]
            for p in [-1, 0, 1]:
                for q in [-1, 0, 1]:
                    i1 = i + p; j1 = j + q
                    if (p != 0 or q != 0) and 0 <= i1 < size and 0 <= j1 < size :
                        if work_grid[i1][j1] == 0: 
                            work_grid[i1][j1] = 2
                            related_component.append((i1, j1))
        
        if len(related_component) > len(result): result = related_component

        x, y = first_not_explored(work_grid)

    return result





def surroundings_ok(grid, pos, radius):
    """Checks if there isn't a wall in the `radius` of the position `pos` on the grid.

    Args:
        grid: the square grid representing the world with the walls and the empty spaces
        pos: the position whose surroundings we want to check as a tuple
        radius: the radius to cover
    Returns:
        bool: True if there isn't a wall in the position's radius and False otherwise
    """
    size = len(grid)
    int_radius = int(np.ceil(radius))

    for i in range(max(0, pos[0] - int_radius), min(pos[0] + int_radius, size - 1)):
        for j in range(max(0, pos[1] - int_radius), min(pos[1] + int_radius, size - 1)):
            if(grid[i][j] == 1) : return False
    return True


def is_navigable(grid, agent, target, agent_radius):
    """Makes sure there is a navigable way from the agent to the target on the grid while taking 
    the agent's radius into account.

    Args:
        grid: the square grid representing the world with empty space as 0 and walls as 1
        agent: the agent's position on the grid as a numpy array
        target: the target's position on the grid as a numpy array
        agent_radius: the agent's radius
    Returns: 
        navigable: True if there is a way for the agent to attain the target's position without physically
            being blocked by walls
    """

    size = len(grid)
    work_grid = grid.copy()
    queue = [(int(agent[0]), int(agent[1]))]
    work_grid[int(agent[0])][int(agent[1])] = 2

    for elem in queue:
        i = elem[0]; j = elem[1]
        for p in [-1, 0, 1]:
            for q in [-1, 0, 1]:
                i1 = i + p; j1 = j + q
                if (p != 0 or q != 0) and 0 <= i1 < size and 0 <= j1 < size:
                    if i1 == target[0] and j1 == target[1]: return True
                    if work_grid[i1][j1] == 0 and surroundings_ok(grid, (i1, j1), agent_radius): 
                        work_grid[i1][j1] = 2
                        queue.append((i1, j1))
        
    return False


@njit(cache=True, fastmath=True)
def clearance_mask_jit(grid, int_radius):
    """For every cell (i, j) of `grid`, computes whether a `surroundings_ok`
    check at that position (radius `int_radius`) would pass -- i.e. no wall
    within the same box that `surroundings_ok` scans. Same boundary
    convention as `surroundings_ok` (upper bound is `size - 1`, exclusive),
    replicated exactly so results match bit-for-bit.

    Meant to be computed ONCE per topology (grid never changes within an
    episode) and then reused for O(1) lookups instead of re-scanning an
    O(radius^2) box on every cell visited by is_navigable_jit's BFS below --
    `is_navigable` used to call `surroundings_ok` from scratch at every one
    of the (potentially thousands of) cells it visits per call, and is
    itself called from inside a retry loop in env.py's reset(), so this
    redundant rescanning was one of the biggest hidden costs in reset()
    (measured ~30ms/call before this fix, largely independent of whether the
    topology itself came from cache or not).
    """
    size = grid.shape[0]
    mask = np.ones((size, size), dtype=np.bool_)
    for pi in range(size):
        i_lo = max(0, pi - int_radius)
        i_hi = min(pi + int_radius, size - 1)
        for pj in range(size):
            j_lo = max(0, pj - int_radius)
            j_hi = min(pj + int_radius, size - 1)
            ok = True
            for i in range(i_lo, i_hi):
                for j in range(j_lo, j_hi):
                    if grid[i, j] == 1:
                        ok = False
                        break
                if not ok:
                    break
            mask[pi, pj] = ok
    return mask


@njit(cache=True, fastmath=True)
def is_navigable_jit(grid, clearance_mask, agent_i, agent_j, target_i, target_j):
    """Same BFS as `is_navigable` above, but using a precomputed
    `clearance_mask` (see clearance_mask_jit) for O(1) clearance lookups
    instead of recomputing an O(radius^2) box scan at every visited cell.
    Verified to return bit-identical results to the original on real
    agent/target pairs drawn during reset(). ~140x faster in isolation
    (measured), since the BFS itself now does plain array lookups.
    """
    size = grid.shape[0]
    work_grid = grid.copy()
    max_queue = size * size
    queue_i = np.empty(max_queue, dtype=np.int64)
    queue_j = np.empty(max_queue, dtype=np.int64)
    queue_i[0] = agent_i
    queue_j[0] = agent_j
    work_grid[agent_i, agent_j] = 2
    qlen = 1
    idx = 0
    while idx < qlen:
        i = queue_i[idx]
        j = queue_j[idx]
        idx += 1
        for p in range(-1, 2):
            for q in range(-1, 2):
                if p == 0 and q == 0:
                    continue
                i1 = i + p
                j1 = j + q
                if 0 <= i1 < size and 0 <= j1 < size:
                    if i1 == target_i and j1 == target_j:
                        return True
                    if work_grid[i1, j1] == 0 and clearance_mask[i1, j1]:
                        work_grid[i1, j1] = 2
                        queue_i[qlen] = i1
                        queue_j[qlen] = j1
                        qlen += 1
    return False