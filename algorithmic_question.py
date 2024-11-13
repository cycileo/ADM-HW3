def algorithm(t, n, grid_list):
    """
    This function processes a list of test cases, where each test case consists of a grid of coordinates. 
    It determines whether it is possible to visit all the coordinates starting from (0, 0), 
    following a valid path (only moving right or up), and outputs the lexicographically smallest path 
    if possible. If it is not possible, it returns "NO".

    Args:
        grid_list (list of list of lists): A list containing multiple test cases. Each test case is represented 
                                            as a list of coordinates, where each coordinate is a list 
                                            of two integers [x, y] representing the position of a package.
                                            The coordinates are given as (x_i, y_i) for each package, 
                                            with 0 <= x_i, y_i <= 100.

    Returns:
        str: A string representing the result for each test case.
              For each test case, it returns either:
              - "YES\n" followed by the lexicographically smallest path (string of "R" for right and "U" for up),
                or 
              - "NO\n" if it's impossible to collect all the packages.
    """

    results = ""  # Initialize an empty string to store results for all test cases
    
    # Loop through each test case
    for i in range(t):
        grid = grid_list[i]  # Get the current grid (list of packages)
        grid.append([0, 0])  # Add the starting point (0, 0) to the grid
        n = len(grid)  # Get the total number of points (including the start)
        
        # Sort the grid by x-coordinate first, then by y-coordinate
        sorted_grid = sorted(grid, key=lambda x: (x[0], x[1]))
        path = ""  # Initialize an empty string to track the path
        
        # Process each consecutive pair of points
        for i in range(0, n-1):
            result = ""  # Initialize the result for the current test case
            
            # Check if the path is valid (cannot go up and left at the same time or down and right)
            if sorted_grid[i][0] < sorted_grid[i+1][0] and sorted_grid[i][1] > sorted_grid[i+1][1]:
                result = "NO\n"  # If the path is invalid, return "NO"
                break
            # If the path is valid
            else:
                # Calculate the steps for the horizontal (right) direction
                if sorted_grid[i][0] < sorted_grid[i+1][0]:
                    n_steps = sorted_grid[i+1][0] - sorted_grid[i][0]  # Number of steps to the right
                    path += "R" * n_steps  # Add 'R' for each step to the right
                
                # Calculate the steps for the vertical (up) direction
                if sorted_grid[i][1] < sorted_grid[i+1][1]:
                    n_steps = sorted_grid[i+1][1] - sorted_grid[i][1]  # Number of steps upwards
                    path += "U" * n_steps  # Add 'U' for each step upwards
                
                # Add "YES" and the calculated steps to the result
                result += "YES\n" + path + "\n"
        
        # Append the result for the current test case to the final results string
        results += result
    
    return results  # Return the final results for all test cases