
def first_not_explored(grid, height, width):
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 0: return (i, j)

    return (-1, -1)

def main_related_component(grid, height, width):
    """A function that takes a 2D grid describing a topology of an environemnt and computes the 
    main related component as a list of positions. We use the convention 0 is empty and 1 is wall.
    
    Args:
        grid: the 2D array representing the topology 
        height: the height of the grid
        width: the width of the grid
    Returns:
        list[tuple[int, int]]: the positions in the main related component
    """

    result = []
    work_grid = grid.copy()

    x, y = first_not_explored(work_grid, height, width)

    while (x, y) != (-1, -1):
        work_grid[x][y] = 2
        related_component = [(x, y)]

        for elem in related_component:
            i = elem[0]; j = elem[1]
            for p in [-1, 0, 1]:
                for q in [-1, 0, 1]:
                    i1 = i + p; j1 = j + q
                    if (p != 0 or q != 0) and i1 >= 0 and i1 < height and j1 >= 0 and j1 < width :
                        if work_grid[i1][j1] == 0: 
                            work_grid[i1][j1] = 2
                            related_component.append((i1, j1))
        
        if len(related_component) > len(result): result = related_component

        x, y = first_not_explored(work_grid, height, width)

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