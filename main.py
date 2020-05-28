import shapefile
import math
import heapq
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os.path
import time


""" constants """
DEFAULT_RESOLUTION = 0.002
DEFAULT_THRESHOLD = 0.5


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
    print("       * * * * * * * * * * * * * * * *\n")

    keep_going = True

    while keep_going:

        # print("       * * * * * * * * * * * * * * * *")
        print("***  Main Menu  ***\n")
        # print("       * * * * * * * * * * * * * * * *\n")
        print("Available range: {-73.59, -73.55)  {45.49, 45.53)")
        print(f"Area Resolution: {resolution}")
        print(f"Crime Threshold: {threshold}")
        print()
        print("{:<18}: enter 'r'".format("Set resolution"))
        print("{:<18}: enter 't'".format("Set threshold"))
        print("{:<18}: enter 'q'".format("Quit"))
        print("\nPress Enter to start!\n")

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
            print("\nBye\n")
            return
        else:
            print("\nInvalid option\n")
            continue


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
    crime_map = Graph(shapes, resolution, threshold)
    fig, ax, title = show_map(crime_map)

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
        user_start = input("\nEnter starting coordinates (ex: -73.55 45.525)\n")
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
            print("\nThe locations must be within the map's range\n")
            input("\nPress enter to continue.\n")
            return

        user_goal = input("\nEnter destination coordinates (ex: -73.55 45.525)\n")
        split_args = user_goal.split()
        goal_longitude, goal_latitude = get_coordinates(*split_args)

        if not goal_longitude or not goal_latitude:
            i -= 1
            print("\nInvalid input. Try again.")
            continue

        goal_node = crime_map.get_graph_node(goal_longitude, goal_latitude)
        if not goal_node:
            print("\nThe locations must be within the map's range\n")
            input("\nPress enter to continue.\n")
            return

        break

    # Calculate heuristic values for the whole graph
    # This could have been done on a need basis, during the A* algorithm,
    # but the data size is small and we hereby avoid calculating heuristic values
    # multiple times for the same node.
    crime_map.add_heuristics(goal_node)

    # A* algorithm
    solution_path = a_star_algorithm(start_node, goal_node, time.time())

    # Show solution
    # start and goal are always shown.
    start_point = crime_map.get_graph_tuple(start_longitude, start_latitude)
    goal_point = crime_map.get_graph_tuple(goal_longitude, goal_latitude)

    show_solution(crime_map, solution_path, fig, ax, title, start_point, goal_point)


class Graph:
    """
    Graph representing the crime map. Each node represents a specific square-area the side length
    of which is determined by the resolution attribute, passed as a parameter.
    Each node has a crime count for its area, as well as tuple representing the geo-location of the
    bottom-left point in the area. Real geo-locations are mapped to a node and take on the coordinates of
    the sub-area it represents.

    Nodes are mapped to a 2-dimensional array to facilitate the building of the graph and the edges. Each node
    a set of edges. Edges contain a reference to a node, as well as a cost value.

    Nodes also have a heuristic value, which is computed once the goal destination is known.
    """

    def __init__(self, shapes, resolution, threshold):

        self.graph = []
        self.resolution = resolution
        self.threshold = threshold

        # costs of edges
        self._cost_along_block = 1.3
        self._cost_diagonal = 1.5
        self._cost_straight = 1

        # total covered area
        self.latitude_min = 45.49
        self.latitude_max = 45.53
        self.longitude_min = -73.59
        self.longitude_max = -73.55

        self._build_graph()
        self._add_crime_stats(shapes)
        self.cutoff_rate = self._place_blocks()
        self._add_edges()

    def _build_graph(self):
        """
        Creates the basic graph of nodes corresponding to all the coordinates in the map, along
        with their respective crime counts.
        :return: None
        """

        # We use this below to avoid floating-point precision issues by temporarily moving the decimal point away
        mov_dec = 10e5
        self.x_axis = np.arange(self.longitude_min * mov_dec,
                                self.longitude_max * mov_dec,
                                self.resolution * mov_dec) / mov_dec
        self.y_axis = np.arange((self.latitude_min * mov_dec),
                                (self.latitude_max * mov_dec),
                                (self.resolution * mov_dec)) / mov_dec

        # adding the basic graph nodes inside the 2D list
        for row in range(len(self.y_axis)):
            self.graph.append([])
            for col in range(len(self.x_axis)):
                point = (self.x_axis[col], self.y_axis[-1 - row])
                self.graph[row].append(GraphNode(point))

    def _add_crime_stats(self, shapes):
        """
        Adds crime counts to appropriate nodes from the data in the shape file.
        :param shapes:
        :return:
        """
        for shape in shapes:
            longitude = shape.points[0][0]
            latitude = shape.points[0][1]
            node = self.get_graph_node(longitude, latitude)
            node.crime_count += 1

    def get_graph_node(self, longitude, latitude):
        """
        Returns a reference to a graph node corresponding
        to a certain geolocation
        :param longitude:
        :param latitude:
        :return: a reference to a graph node
        """
        mov_dec = 10e5  # this is used to avoid floating point accuracy issues
        x = math.floor((longitude * mov_dec - self.longitude_min * mov_dec) / (self.resolution * mov_dec))
        y = len(self.graph) - 1 - \
            math.floor((latitude * mov_dec - self.latitude_min * mov_dec) / (self.resolution * mov_dec))

        # checking that the geolocation is reachable in our graph
        if not (0 <= x < len(self.graph[0]) and 0 <= y < len(self.graph)):
            return None

        return self.graph[y][x]

    def get_graph_tuple(self, longitude, latitude):
        """
        Returns the internal indices associated with a node
        corresponding to a geo-location.
        :param longitude:
        :param latitude:
        :return: a tuple representing the indices i, j of a node location in the 2-dimensional list
        """
        mov_dec = 10e5  # this is used to avoid floating point accuracy issues
        x = math.floor((longitude * mov_dec - self.longitude_min * mov_dec) / (self.resolution * mov_dec))
        y = len(self.graph) - 1 - \
            math.floor((latitude * mov_dec - self.latitude_min * mov_dec) / (self.resolution * mov_dec))

        # checking that the geolocation is reachable in our graph
        if not (0 <= x < len(self.graph[0]) and 0 <= y < len(self.graph)):
            return None

        return x, y

    def _place_blocks(self):
        """
        Determines which nodes will be a block according to the supplied threshold parameter.
        :return: None
        """

        # list flattening, to facilitate the application of a threshold
        # and determine which nodes will be blocks
        flat_list = [node for row in self.graph for node in row]
        flat_list.sort(key=lambda cell: cell.crime_count, reverse=True)

        # apply threshold rate
        num_blocks = math.floor((1 - self.threshold) * len(flat_list))

        if num_blocks < 1:  # there are no blocks
            return math.inf

        # update cells above the threshold to indicate they are blocks
        for index in range(num_blocks):
            flat_list[index].block = True

        # also include cells that have same crime rate as the last node included
        # if any. This allows straight forward identification in the
        # color plot.
        cutoff_rate = flat_list[index].crime_count
        index += 1
        while index < len(flat_list) and flat_list[index].crime_count >= cutoff_rate:
            flat_list[index].block = True
            index += 1

        return cutoff_rate

    def add_heuristics(self, goal_node):
        """
        Calculates and sets the heuristic value for every node in the graph. See
        the heuristic function for more details.
        :param goal_node: destination node
        :return: None
        """
        for row in self.graph:
            for node in row:
                node.heuristic = self.heuristic(node, goal_node)

    def heuristic(self, node, goal_node):
        """
        Returns the shortest absolute distance, in terms of resolution units (see resolution attribute),
        between a node and the goal. It uses Pythagoras formula to get the absolute distance between
        a point and the goal, and then divides the result by the resolution (or the sub-area side size) to
        put the result in the same units as the path cost:
        [( (y_goal - y)^2 + (x_goal - x)^2 )^(1/2)] / resolution.

        This heuristic is admissible and monotonous.

        ADMISSIBILITY
        It is admissible because it calculates the absolute shortest distance between a point and the goal.
        Therefore, it will never overestimate the true cost of a path. The real cost is limited to straight and diagonal
        moves from one node to the other, ascribing 1 or 1.3 units to straight moves and 1.5 to diagonal moves, and
        cannot go through blocks.
        Instead, this heuristic will return the absolute minimum value in terms of resolution units (1 for straight
        moves, 1.414 (or the square-root of 2) for immediate diagonals, and even more optimistic measures for farther
        distances,because the measures allow to cut across segments).

        MONOTONICITY
        For the monotonous aspect, it follows from the previous description that,
        for cell x1 and neighbour x2, h(x1) <= cost(x, x2) + h(x2), since no move to a reachable node
        will ever improve the distance returned by the heuristic function, since it is already the shortest possible.

        For these reasons, the goal will always be reached by the fastest route
        and each node will always be visited at the lowest cost the first time they are encountered.n

        INFORMEDNESS
        This heuristic function is reasonably informed, taking into account the true distance to the target.
        It could have been perhaps more informed, but this would have been costlier and error-prone: taking into account
        the presence of blocks would have make it easier to design a heuristic that is not admissible or monotonous,
        because of corner cases and other subtleties. Such a cost was not worth it. The focus was on deigning an
        algorithm that works efficiently.
        Furthermore, the A* algorithm is tasked with taking the blocks
        into account, so there is no need to try to do that work separately, on our own.

        :param node: the node for which we want to calculate the heuristic value
        :param goal_node: the goal node, the position of which determines the heuristic value of the node
        :return:
        """
        y_diff = goal_node.point[1] - node.point[1]
        x_diff = goal_node.point[0] - node.point[0]
        # the division by the sub-grid size is to put distances in the same units as the length cost
        return math.sqrt(y_diff ** 2 + x_diff ** 2) / self.resolution

    def get_statistics(self):
        """
        Obtains the mean and standard deviation for crime stats of this graph.
        :return: a tuple of the mean and the standard deviation for crime stats of this graph.
        """
        crime_stats = [node.crime_count for row in self.graph for node in row]
        mean = stats.mean(crime_stats)
        std_dev = stats.stdev(crime_stats)
        return mean, std_dev

    def _add_edges(self):
        """
        Discovers all edges for each cell in the graph, according to the blocks
        and cost rules. Should only be called once blocks have been assigned.
        Edges along the boundaries of the graph aren't allowed.
        :return: None
        """
        for i in range(len(self.graph)):
            for j in range(len(self.graph[0])):
                self._get_edges(i, j)

    def _get_edges(self, i, j):
        """
        Get edges for one node. Should only be called once blocks have been assigned.
        :param i: the row index of the node in the 2D array
        :param j: the colomn index of the node in the 2D array
        :return: None
        """
        node = self.graph[i][j]

        # East-bound edges
        if j < len(self.graph[0]) - 1:
            # if top-row
            if i == 0:
                if node.block and self.graph[i + 1][j].block:
                    pass  # no east edges
                elif node.block:
                    node.edges.extend([Edge(self.graph[i][j + 1], self._cost_along_block),
                                       Edge(self.graph[i + 1][j + 1], self._cost_diagonal)])
                elif self.graph[i + 1][j].block:
                    node.edges.append(Edge(self.graph[i][j + 1], self._cost_along_block))
                else:
                    node.edges.extend([Edge(self.graph[i][j + 1], self._cost_straight),
                                       Edge(self.graph[i + 1][j + 1], self._cost_diagonal)])
            # if bottom-row
            elif i == len(self.graph) - 1:
                if not node.block:
                    node.edges.append(Edge(self.graph[i - 1][j + 1], self._cost_diagonal))
            # regular nodes
            else:
                if node.block and self.graph[i + 1][j].block:
                    pass  # no east edges
                elif node.block:
                    node.edges.extend([Edge(self.graph[i][j + 1], self._cost_along_block),
                                       Edge(self.graph[i + 1][j + 1], self._cost_diagonal)])
                elif self.graph[i + 1][j].block:
                    node.edges.extend([Edge(self.graph[i - 1][j + 1], self._cost_diagonal),
                                       Edge(self.graph[i][j + 1], self._cost_along_block)])
                else:
                    node.edges.extend([Edge(self.graph[i - 1][j + 1], self._cost_diagonal),
                                       Edge(self.graph[i][j + 1], self._cost_straight),
                                       Edge(self.graph[i + 1][j + 1], self._cost_diagonal)])

        # West-bound edges
        if j > 0:
            # if top-row
            if i == 0:
                if self.graph[i][j - 1].block and self.graph[i + 1][j - 1].block:
                    pass  # no west edges
                elif self.graph[i][j - 1].block:
                    node.edges.extend([Edge(self.graph[i][j - 1], self._cost_along_block),
                                       Edge(self.graph[i + 1][j - 1], self._cost_diagonal)])
                elif self.graph[i + 1][j - 1].block:
                    node.edges.append(Edge(self.graph[i][j - 1], self._cost_along_block))
                else:
                    node.edges.extend([Edge(self.graph[i][j - 1], self._cost_straight),
                                       Edge(self.graph[i + 1][j - 1], self._cost_diagonal)])
            # if bottom-row
            elif i == len(self.graph) - 1:
                if not node.block:
                    node.edges.append(Edge(self.graph[i - 1][j - 1], self._cost_diagonal))
            # regular nodes
            else:
                if self.graph[i][j - 1].block and self.graph[i + 1][j - 1].block:
                    pass  # no east edges
                elif self.graph[i][j - 1].block:
                    node.edges.extend([Edge(self.graph[i][j - 1], self._cost_along_block),
                                       Edge(self.graph[i + 1][j - 1], self._cost_diagonal)])
                elif self.graph[i + 1][j - 1].block:
                    node.edges.extend([Edge(self.graph[i - 1][j - 1], self._cost_diagonal),
                                       Edge(self.graph[i][j - 1], self._cost_along_block)])
                else:
                    node.edges.extend([Edge(self.graph[i - 1][j - 1], self._cost_diagonal),
                                       Edge(self.graph[i][j - 1], self._cost_straight),
                                       Edge(self.graph[i + 1][j - 1], self._cost_diagonal)])

        # north-bound edge
        if i > 0 and j > 0:
            if node.block and self.graph[i][j - 1].block:
                pass
            elif node.block or self.graph[i][j - 1].block:
                node.edges.append(Edge(self.graph[i - 1][j], self._cost_along_block))
            else:  # no blocks
                node.edges.append(Edge(self.graph[i - 1][j], self._cost_straight))

        # south-bound edge
        if i < len(self.graph) - 1 and j > 0:
            if self.graph[i + 1][j].block and self.graph[i + 1][j - 1].block:
                pass
            elif self.graph[i + 1][j].block or self.graph[i + 1][j - 1].block:
                node.edges.append(Edge(self.graph[i + 1][j], self._cost_along_block))
            else:  # no blocks
                node.edges.append(Edge(self.graph[i + 1][j], self._cost_straight))


class GraphNode:
    """
    Represents a square area within the original map.
    Contains crime statistics for the area. Has a tuple attribute representing the
    bottom-left geo-locations of the area, that all locations in this area are mapped to.
    The boolean block indicates whether the node represents a block on the map.
    due to a crime_count above the threshold
    It also has a list of Edge objects representing edges to reachable nodes
    and their associated cost.
    """

    def __init__(self, point):
        self.point = point
        self.heuristic = 0    # heuristic value
        self.crime_count = 0
        self.block = False
        self.edges = []


class Edge:
    """
    Represents an edge from one GraphNode to another
    with the associated cost.
    Edges are only discovered once a threshold has been established
    and blocks on the maps have been located.
    """
    def __init__(self, graph_node, cost):
        self.graph_node = graph_node
        self.cost = cost


class StateSpaceNode:
    """
    Represents a node in the A* algorithm implementation.
    Contains a reference to a GraphNode, as well as
    information pertaining to the search algorithm.
    It has a parent node (to retrace the solution path), and a set of child nodes
    representing the nodes reachable from this node. It has a total path cost,
    which is the cost from the start node to this node, according to the specific
    branch in the search tree, as well as an A* score, which is the sum of the total path
    cost and the related graph node's heuristic value.
    """
    def __init__(self, graph_node, parent, path_cost, a_star_score):
        self.graph_node = graph_node
        self.parent = parent
        self.children = []
        self.path_cost = path_cost
        self.a_star_score = a_star_score

    def __lt__(self, other):
        return self.a_star_score < other.a_star_score


def a_star_algorithm(starting_node, goal_node, start_time):
    """
    Since the heuristic function is admissible (see relevant comments), when we
    encounter the goal node, the solution path will also be the most effective path.
    Also, since the heuristic function is monotonous (see relevant comments),
    we store visited nodes in a set, since they will always be visited only once,
    at the best possible cost.
    :param starting_node:
    :param goal_node:
    :param start_time:
    :return: the solution path as a list of tuples
    """

    solution_path = []    # a list of points from the goal back to the root of the tree
    open_list = []        # priority queue. The key is the A* score, and the value is a node reference
    closed_list = {None}  # this is the set (see comments above) of tuples representing the geo-locations of the visited nodes.

    root = StateSpaceNode(starting_node, None, 0, starting_node.heuristic)
    heapq.heappush(open_list, root)

    while open_list:

        # checking timing requirement
        if time.time() - start_time > 10.0:
            print("Time is up. The optimal path is not found.")
            input("Press enter to terminate program.")
            exit()

        node = heapq.heappop(open_list)

        if node.graph_node is goal_node:
            # we have reached the goal
            solution_path = get_solution_path(node)
            return solution_path

        closed_list.add(node.graph_node.point)

        for edge in node.graph_node.edges:
            if edge.graph_node.point in closed_list:
                continue  # ignore a visited node
            total_cost = node.path_cost + edge.cost
            a_star_score = total_cost + edge.graph_node.heuristic
            tree_node = StateSpaceNode(edge.graph_node, node, total_cost, a_star_score)
            heapq.heappush(open_list, tree_node)     # adding new node to the open list, with A* score as key

    # if we ended up here, it means we have exhausted all paths
    # but the search yielded no solution
    return solution_path


def get_solution_path(node):
    """
    Reconstruct the path from the root node to the goal node
    :param node:
    :return: a list of tuples representation geo-locations
    """
    solution_path = [node.graph_node.point]
    while node.parent:
        node = node.parent
        solution_path.append(node.graph_node.point)

    return solution_path


def show_map(crime_map):
    """
    Shows the initial graph, before specifying a start and a goal
    :param crime_map:
    :return:
    """

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
    bounds = [0, crime_map.cutoff_rate, 3000]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7))
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
    y_ticks = np.arange(crime_map.latitude_max * mov_dec,
                        (crime_map.latitude_min - crime_map.resolution) * mov_dec,
                        -crime_map.resolution * mov_dec) / mov_dec

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

    ax.scatter(start_point[0] - 0.5, start_point[1] + 0.5, color='green', linewidths=5, zorder=4,
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
        message = f"{title}\nPath found! See path on map. Length: {len(solution_path) - 1}"
        if len(solution_path) == 1:
            message += "  - Goal is located in same area as starting point"

        # convert path
        solution_x = []
        solution_y = []
        for i in range(len(solution_path) - 1, -1, -1):
            point = crime_map.get_graph_tuple(*solution_path[i])
            solution_x.append(point[0] - 0.5)
            solution_y.append(point[1] + 0.5)

        ax.plot(solution_x, solution_y, color='green', linewidth=4, zorder=2)
        fig.suptitle(message)
        print(f"\n{message}\n")
        print(solution_path)
        print()
        plt.show(block=False)

    input("\nPress enter to continue\n")


def set_resolution():
    """
    Determines the side length of sub square-areas in the crime map
    for a map to be built.
    :return: None
    """
    i = 3
    while i > 0:
        try:
            res = float(input("\nEnter resolution between 0.0015 and 0.005\n"))

        except ValueError:
            print("\nInvalid resolution\n")
            i -= 1
        else:
            if not 0.0015 <= res <= 0.005:
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


if __name__ == "__main__":
    main()
