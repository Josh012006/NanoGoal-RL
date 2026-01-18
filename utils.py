import numpy as np

def first_not_explored(grid):
    """Finds the first empty cell in the grid that wasn't already explored. Note that 1 is for wall,
    0 is for empty space not explored and 2 is for space already explored.
    Args:
        grid: the suare grid to work on
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
    main related component as a list of positions. We use the convention 0 is empty and 1 is wall.
    
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

# grid = [
#     [1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 1, 0, 1],
#     [1, 0, 0, 1, 0, 1],
#     [1, 1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1],
# ]

# print(main_related_component(grid, 6, 6))



def surroundings_ok(grid, pos, radius):
    """Checks if there isn't a wall in the radius of the position `pos` on the grid.

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