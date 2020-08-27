# -------------------------------------------------------
# Assignment 1
# Written by Simon Brillant-Giroux, 40089110
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

import numpy as np
import os.path
import shapefile
import matplotlib.pyplot as plt
from matplotlib import colors

from a_star_algorithm import a_star_algorithm
from crime_rate_graph import Graph

""" constants """
DEFAULT_RESOLUTION = 0.002
DEFAULT_THRESHOLD = 0.75

LATITUDE_MIN = 45.49
LATITUDE_MAX = 45.53
LONGITUDE_MIN = -73.59
LONGITUDE_MAX = -73.55


def main():
    """
    Deals with user interaction. Handles the flow of the program.
    :return: None
    """
    sf = shapefile.Reader(os.path.join("Shape", "crime_dt.shp"))
    shapes = sf.shapes()
    resolution = DEFAULT_RESOLUTION
    threshold = DEFAULT_THRESHOLD

    print("\n")
    print("       * * * * * * * * * * * * * * * *")
    print("       *  Shortest Safe Path Finder  *")
    print("       * * * * * * * * * * * * * * * *")

    keep_going = True

    while keep_going:

        print("\n***  Main Menu  ***\n")

        print("\n")
        print("* * * * * * * *")
        print("*  Settings   *")
        print("* * * * * * * *\n")

        print(f"Available range: LON: [{LONGITUDE_MIN}, {LONGITUDE_MAX}[  LAT: [{LATITUDE_MIN}, {LATITUDE_MAX}[")
        print(f"Area Resolution: {resolution}")
        print(f"Crime Threshold: {threshold}")

        print("\n")
        print("* * * * * * * * * *")
        print("*  Options Menu   *")
        print("* * * * * * * * * *\n")

        print("{:<18}: enter 'r'".format("Set resolution"))
        print("{:<18}: enter 't'".format("Set threshold"))
        print("{:<18}: enter 'q'".format("Quit"))
        print("\n{:<18}: press Enter!\n".format("Start"))

        user_input = input().strip()

        if user_input is '':
            start(shapes, resolution, threshold)
            continue
        elif user_input is 'r':
            temp = set_resolution()
            if temp:
                resolution = temp
        elif user_input is 't':
            temp = set_threshold()
            if temp:
                threshold = temp
        elif user_input is 's':
            pass
        elif user_input is 'q':
            print("\nBye, thanks for using Path Finder!\n")
            return
        else:
            print("\nInvalid option\n")
            continue


def set_resolution():
    """
    Determines the side length of sub square-areas in the crime map
    for a map to be built.
    :return: None
    """
    i = 3
    while i > 0:
        try:
            res = float(input("\nEnter resolution between 0.001 and 0.005 (for a better experience: 0.002 or above)\n"))

        except ValueError:
            print("\nInvalid resolution\n")
            i -= 1
        else:
            if not 0.001 <= res <= 0.005:
                i -= 1
                print("\nValue not in range.\n")
                continue
            else:
                return res

    return None


def set_threshold():
    """
    Sets the threshold at which sub-areas will be considered as blocks.
    Must be a value between 1% to 100%
    :return: None
    """
    i = 3  # max of three attempts
    while i > 0:
        try:
            thr = int(input("\nEnter threshold % (1 - 100)\n"))

        except ValueError:
            print("\nInvalid format\n")
            i -= 1
        else:
            if not 1 <= thr <= 100:
                i -= 1
                print("\nValue not in range\n")
                continue
            else:
                return thr / 100.0

    return None


def start(shapes, resolution, threshold):
    """
    Gets start and goal locations from user, shows the original map,
    and provides the most effective path
    to the destination on the terminal and on the graphic map.
    Uses an A* algorithm, with an admissible and monotonous heuristic function.
    More information at the appropriate methods and class definitions.
    :param shapes: the shapefile object obtained from shape files, using the PyShp library
    :param resolution: the size of the side of each sub-areas in hte map (grids), in terms of geo-location units.
    :param threshold: The percentage of the sub-areas (arranged from the crime to the highest crime rates),
    from which we start to label sub-areas as blocks (forbidden areas)
    :return:
    """

    # Build and show the map
    crime_map = Graph(shapes, resolution, threshold, LATITUDE_MIN, LATITUDE_MAX, LONGITUDE_MIN, LONGITUDE_MAX)
    fig, ax, title = show_map(crime_map)

    go = input("\nPress 'r' to reconfigure the map,\nPress Enter to continue with current map:\n")
    while go is not 'r' and go is not '':
        go = input("Press Enter to continue, or 'r' to reconfigure the map")

    if go is 'r':
        plt.close()
        return

    # get start and goal coordinates from user
    start_longitude = 0
    start_latitude = 0
    goal_longitude = 0
    goal_latitude = 0

    def get_coordinates(x, y):
        try:
            longitude = float(x)
            latitude = float(y)
        except ValueError:
            return None, None
        else:
            return longitude, latitude

    # Checking validity of input, with a max of three attempts.
    i = 3
    while i > 0:
        user_start = input("\nEnter starting longitude and latitude, separated by a space (ex: -73.56 45.525)\n")
        split_args = user_start.split()
        if len(split_args) != 2:
            print("\nInvalid input. Try again.")
            i -= 1
            continue
        start_longitude, start_latitude = get_coordinates(*split_args)

        if not start_longitude or not start_latitude:
            i -= 1
            print("\nInvalid input. Try again.")
            continue

        start_node = crime_map.get_graph_node(start_longitude, start_latitude)
        if not start_node:
            i -= 1
            print("\nThe locations must be within the map's range\n")
            input("\nPress enter to continue.\n")
            continue

        user_goal = input("\nEnter destination longitude and latitude, separated by a space (ex: -73.56 45.525)\n")
        split_args = user_goal.split()
        if len(split_args) != 2:
            print("\nInvalid input. Try again.")
            i -= 1
            continue
        goal_longitude, goal_latitude = get_coordinates(*split_args)

        if not goal_longitude or not goal_latitude:
            i -= 1
            print("\nInvalid input. Try again.")
            continue

        goal_node = crime_map.get_graph_node(goal_longitude, goal_latitude)
        if not goal_node:
            i -= 1
            print("\nThe locations must be within the map's range\n")
            input("\nPress enter to continue.\n")
            continue

        break

    if i == 0:  # three failed attempts: return to main menu
        return

    # Calculate heuristic values for the whole graph
    # This could have been done on a need basis, during the A* algorithm,
    # but the data size is small and we hereby avoid calculating heuristic values
    # multiple times for the same node.
    crime_map.add_heuristics(goal_node)

    # A* algorithm
    solution_path = a_star_algorithm(start_node, goal_node)

    # Show solution
    # start and goal are always shown.
    start_point = crime_map.get_graph_tuple(start_longitude, start_latitude)
    goal_point = crime_map.get_graph_tuple(goal_longitude, goal_latitude)

    show_solution(crime_map, solution_path, fig, ax, title, start_point, goal_point)


def show_map(crime_map):
    """
    Shows the initial graph, before specifying a start and a goal
    :param crime_map:
    :return:
    """

    # defining figure-size
    figure_size = (9, 7) if crime_map.resolution > 0.00175 else (11, 11)

    # remove any previous plot
    plt.close()

    # get statistics to display
    mean, std_dev = crime_map.get_statistics()

    # getting a 2D list of the crime counts to map to a colormap
    data = np.zeros((len(crime_map.graph[0]), len(crime_map.graph)))
    for row in range(len(crime_map.graph)):
        for col in range(len(crime_map.graph[0])):
            data[row, col] = crime_map.graph[row][col].crime_count

    # create discrete colormap
    cmap = colors.ListedColormap(['purple', 'yellow'])
    bounds = [0, crime_map.cutoff_rate - 0.1, 3000]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figure_size)
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw grid lines
    ax.grid(b=True, which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-0.5, len(crime_map.graph[0]), 1))
    ax.set_yticks(np.arange(-0.5, len(crime_map.graph), 1))

    for (i, j), z in np.ndenumerate(data):  # row, column
        ax.text(j, i, int(z), ha='center', va='center')  # col, row, show crime counts for each sub-area

    # relabelling the x, y ticks according to appropriate geo-locations
    mov_dec = 10e5   # this is to avoid floating point accuracy issues in below calculations
    x_ticks = np.arange(crime_map.longitude_min * mov_dec,
                        (crime_map.longitude_max + crime_map.resolution) * mov_dec,
                        crime_map.resolution * mov_dec) / mov_dec
    y_ticks = np.arange(crime_map.latitude_min * mov_dec,
                        (crime_map.latitude_max + crime_map.resolution) * mov_dec,
                        crime_map.resolution * mov_dec) / mov_dec
    # reverse to start the ticks from the bottom
    y_ticks = y_ticks[::-1]

    ax.set_yticklabels(y_ticks)
    ax.set_xticklabels(x_ticks, rotation='vertical')

    title = "Crime counts per area\nMean: {}    Standard deviation: {:0.2f}".format(mean, std_dev)
    fig.suptitle(title)
    plt.show(block=False)

    print("\nMean: {}    Standard deviation: {:0.2f}\nSee map for more information.\n".format(mean, std_dev))

    return fig, ax, title


def show_solution(crime_map, solution_path, fig, ax, title, start_point, goal_point):
    """
    Showing the same graph, but with the solution path
    :param crime_map: the crime map (Graph)
    :param solution_path: a list of tuples representing the path from start to goal.
    :param fig: a reference to the previously obtained pyplot.figure object
    :param ax: a reference to the previously obtained matplotlib.axes.Axes object
    :param title: title of the plot taken from the previously shown plot
    :param start_point:
    :param goal_point:
    :return: None
    """

    ax.scatter(start_point[0] - 0.5, start_point[1] + 0.5, color='orange', linewidths=5, zorder=4,
               label="starting point")
    ax.scatter(goal_point[0] - 0.5, goal_point[1] + 0.5, color='blue', linewidths=9, zorder=3,
               label="goal")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), shadow=True, ncol=2)

    if not solution_path:
        message = f"{title}\nDue to blocks, no solution was found. Try again with other parameters.\n"
        fig.suptitle(message)
        print(f"\n{message}\n")
        plt.show(block=False)

    else:
        # draw the path from the solution_path
        # should work also for len(path) == 1
        message = f"{title}\nPath found! Length: {len(solution_path) - 1}"
        if len(solution_path) == 1:
            message += "  - Goal is located in same area as starting point"

        # convert path
        solution_x = []
        solution_y = []
        for i in range(len(solution_path) - 1, -1, -1):
            point = crime_map.get_graph_tuple(*solution_path[i])
            solution_x.append(point[0] - 0.5)
            solution_y.append(point[1] + 0.5)

        line_width = 4 if crime_map.resolution >= 0.0015 else 3

        ax.plot(solution_x, solution_y, color='green', linewidth=line_width, zorder=2)
        fig.suptitle(message)
        print(f"\n{message}\n")
        print(solution_path)
        print("\nSee path on map!\n")
        plt.show(block=False)

    input("\nPress enter to continue\n")


if __name__ == "__main__":
    main()
