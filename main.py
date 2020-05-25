import shapefile
import math
import heapq
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np



# constants
DEFAULT_RESOLUTION = 0.002
DEFAULT_THRESHOLD = 0.5


# Main loop, dealing with user interaction
def main():
    sf = shapefile.Reader("Shape/crime_dt.shp")
    shapes = sf.shapes()

    print("\n")
    print("       * * * * * * * * * * * * * * * *")
    print("       *  Shortest Safe Path Finder  *")
    print("       * * * * * * * * * * * * * * * *\n")

    resolution = DEFAULT_RESOLUTION
    threshold = DEFAULT_THRESHOLD
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
            specify_and_run(shapes, resolution, threshold)
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


def specify_and_run(shapes, resolution, threshold):

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

    # Calculate heuristics for the whole graph
        # We could have done it on-the-fly, but the limited size of our sample allows this.
        # Also, this prevents calculating the heuristic multiple times for the same node
    crime_map.add_heuristics(goal_node)

    # A* algorithm
    solution_path = a_star_algorithm(start_node, goal_node)

    # Show solution
    # start and goal are shown no matter what
    start_point = crime_map.get_graph_tuple(start_longitude, start_latitude)
    goal_point = crime_map.get_graph_tuple(goal_longitude, goal_latitude)

    show_solution(crime_map, solution_path, fig, ax, title, start_point, goal_point)


# Graph of nodes represent geo-locations within a pre-defined area,
# located in a grid
# The nodes contain crime statistics taken from
class Graph:

    def __init__(self, shapes, resolution, threshold):

        self.graph = []
        self.resolution = resolution
        self.threshold = threshold

        # costs of edges
        self._cost_along_block = 1.3
        self._cost_diagonal = 1.5
        self._cost_straight = 1

        # covered area
        self.latitude_min = 45.49
        self.latitude_max = 45.53
        self.longitude_min = -73.59
        self.longitude_max = -73.55

        self._build_graph()
        self._add_crime_stats(shapes)
        self.cutoff_rate = self._place_blocks()
        self._add_edges()

    # create the basic nodes corresponding to all the coordinates in our graph
    def _build_graph(self):

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

    # adding crime stats to appropriate nodes from the shape file
    def _add_crime_stats(self, shapes):
        for shape in shapes:
            longitude = shape.points[0][0]
            latitude = shape.points[0][1]
            node = self.get_graph_node(longitude, latitude)
            node.crime_count += 1

    # returns a reference to a graph node corresponding
    # to a certain geolocation
    def get_graph_node(self, longitude, latitude):
        mov_dec = 10e5  # this is used to avoid floating point accuracy issues
        x = math.floor((longitude * mov_dec - self.longitude_min * mov_dec) / (self.resolution * mov_dec))
        y = len(self.graph) - 1 - \
            math.floor((latitude * mov_dec - self.latitude_min * mov_dec) / (self.resolution * mov_dec))

        # checking that the geolocation is reachable in our graph
        if not (0 <= x < len(self.graph[0]) and 0 <= y < len(self.graph)):
            return None

        return self.graph[y][x]

    def get_graph_tuple(self, longitude, latitude):
        mov_dec = 10e5  # this is used to avoid floating point accuracy issues
        x = math.floor((longitude * mov_dec - self.longitude_min * mov_dec) / (self.resolution * mov_dec))
        y = len(self.graph) - 1 - \
            math.floor((latitude * mov_dec - self.latitude_min * mov_dec) / (self.resolution * mov_dec))

        # checking that the geolocation is reachable in our graph
        if not (0 <= x < len(self.graph[0]) and 0 <= y < len(self.graph)):
            assert False
            print("rats")
            return None

        return x, y

    # determine which nodes will be a block and label accordingly
    def _place_blocks(self):

        # list flattening, to facilitate the application of a threshold
        # and determine which nodes will be blocks
        flat_list = [node for row in self.graph for node in row]
        flat_list.sort(key=lambda cell: cell.crime_count, reverse=True)

        # apply threshold rate
        num_blocks = math.floor((1 - self.threshold) * len(flat_list))

        # update cells above the threshold to indicate they are blocks
        for index in range(num_blocks):
            flat_list[index].block = True

        # also include cells that have same crime rate as median
        # if that is the case
        cutoff_rate = flat_list[index].crime_count
        index += 1
        while flat_list[index].crime_count >= cutoff_rate:
            flat_list[index].block = True
            index += 1

        return cutoff_rate

    def add_heuristics(self, goal_node):
        for row in self.graph:
            for node in row:
                node.heuristic = self.heuristic(node, goal_node)

    def heuristic(self, node, goal_node):
        """
        Returns the shortest distance, in terms of resolution units (see resolution attribute),
        between a node and the goal.
        This heuristic is admissible and monotonous.
        It is admissible because the distance is calculated with Pythagoras theorem.

        Therefore, it will never overestimate the true cost of a path, which is limited to straight and diagonal
        moves from one node to the other, and ascribes 1 or 1.3 units to straight moves and 1.5 to diagonal moves.
        Instead, it will return the absolute minimum value in terms of resolution units (1 for straight moves, 1.414
        for immediate diagonals, and much more optimistic measures for farther distances).

        It follows that, for the monotonous aspect, we have, for cell x and neighbour x2, h(x) <= cost(x, x2) + h(x2).

        For these reasons, the goal will always be reached by the fastest route
        and each node will always be visited at the lowest cost the the first time they are encountered.
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
        Obtain mean and standard deviation for crime stats of this graph.
        :return: a tuple of the mean and the standard deviation for crime stats of this graph.
        """
        crime_stats = [node.crime_count for row in self.graph for node in row]
        mean = stats.mean(crime_stats)
        std_dev = stats.stdev(crime_stats)
        return mean, std_dev

    def _add_edges(self):
        """
        Obtain edges for each cell in the graph.
        Edges along the boundaries of the graph aren't allowed.
        :return: None
        """

        for i in range(len(self.graph)):
            for j in range(len(self.graph[0])):
                self._get_edges(i, j)

    def _get_edges(self, i, j):
        """
        Get edges for a node. Should only be called once blocks have been assigned.
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


# represents a square area within the original map
# and contains crime statistics for the area
# Dimension is determined by the resolution
# the boolean block means that the area will be represented as block
# due to a crime_count above the threshold
# It also contains a list of Edge objects representing edges to reachable nodes
# and the associated cost. This list will be useful for the implementation of the A* algorithm
class GraphNode:

    def __init__(self, point):
        self.point = point
        self.heuristic = 0    # heuristic value
        self.crime_count = 0
        self.block = False
        self.edges = []


# represents an edge from one GraphNode to another
# with the associated cost
# Edges are discovered only once a threshold has been established
# and blocks on the maps have been located
class Edge:
    def __init__(self, graph_node, cost):
        self.graph_node = graph_node
        self.cost = cost


# represents a node in the A* algorithm implementation
# contains a reference to a graph node
# as well as informative pertaining to the search algorithm
# such as children, parent node, total path cost, and A* score
class StateTreeNode:
    def __init__(self, graph_node, parent, path_cost, a_star_score):
        self.graph_node = graph_node
        self.parent = parent
        self.children = []
        self.path_cost = path_cost
        self.a_star_score = a_star_score

    def __lt__(self, other):
        return self.a_star_score < other.a_star_score


# Since the heuristic function is admissible, when we encounter the goal node, the solution path will
# also be the most effective path
# Note: since the heuristic function is monotonous, we store visited nodes in a set, since they will
# always be visited at the best cost the first time.
def a_star_algorithm(starting_node, goal_node):

    solution_path = []    # a list of points from the goal back to the root of the tree
    open_list = []        # priority queue, key is A* score, value is tree node reference
    closed_list = {None}  # this is a set because the heuristic function is monotonous

    root = StateTreeNode(starting_node, None, 0, starting_node.heuristic)
    heapq.heappush(open_list, root)

    while open_list:

        #todo chck elapsed time

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
            tree_node = StateTreeNode(edge.graph_node, node, total_cost, a_star_score)
            # adding new node to the open list, with A* score as key
            heapq.heappush(open_list, tree_node)

    # if we ended up here, it means we have exhausted all paths
    # but the search yielded no solution
    return solution_path


def get_solution_path(node):
    solution_path = [node.graph_node.point]
    while node.parent:
        node = node.parent
        solution_path.append(node.graph_node.point)

    return solution_path


def show_map(crime_map):

    # remove any previous plot
    plt.close()

    # get statistics
    mean, std_dev = crime_map.get_statistics()

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
        ax.text(j, i, int(z), ha='center', va='center')  # col, row

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

    print("\nSee map for crime stats.\nMean: {}    Standard deviation: {:0.2f}\n".format(mean, std_dev))

    return fig, ax, title


def show_solution(crime_map, solution_path, fig, ax, title, start_point, goal_point):

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
        # draw path from path[]
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

        ax.plot(solution_x, solution_y, color='green', linewidth=5, zorder=2)
        fig.suptitle(message)
        print(f"\n{message}\n")
        print(solution_path)
        print()
        plt.show(block=False)

    input("\nPress enter to continue\n")


def set_resolution():
    i = 3
    while i > 0:
        try:
            res = float(input("\nEnter resolution between 0.001 and 0.005\n"))

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
    i = 3
    while i > 0:
        try:
            thr = int(input("\nEnter threshold % (1 - 99)\n"))

        except ValueError:
            print("\nInvalid format\n")
            i -= 1
        else:
            if not 1 <= thr <= 99:
                i -= 1
                print("\nValue not in range\n")
                continue
            else:
                return thr / 100.0

    return None


if __name__ == "__main__":
    main()
