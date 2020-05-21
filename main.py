import shapefile
import math
import heapq
import statistics as stats
import matplotlib.pyplot as plt


# constants
LATITUDE_RANGE = (45.49, 45.53)
LONGITUDE_RANGE = (-73.59, -73.55)
GRID_SIZE = 0.002 # size of square areas represented by nodes

COST_ALONG_BLOCK = 1.3
COST_DIAGONAL = 1.5
COST_STRAIGHT = 1


def main():

    # preparation
    input("Press to load")
    sf = shapefile.Reader("Shape/crime_dt.shp")
    shapes = sf.shapes()

    input("Press to build")
    # Build a graph with a 2d list representing geolocations of the square areas
    # For now, the edge lists are empty.
    graph = build_graph(shapes)

    # getting a flat descending order list of references to our graph cells
    flat_list = [node for row in graph for node in row]
    flat_list.sort(key=lambda cell: cell.crime_count, reverse=True)

    # get statistics
    crime_stats = [node.crime_count for row in graph for node in row]
    mean = stats.mean(crime_stats)
    std_dev = stats.stdev(crime_stats)
    print(f"mean: {mean}")
    print(f"standard deviation: {std_dev}")
    del crime_stats

    # specific set-up

    # apply threshold rate
    threshold = 0.9
    num_blocks = math.floor((1 - threshold) * len(flat_list))

    # update cells above the threshold to indicate they are blocks
    for index in range(num_blocks):
        flat_list[index].block = True

    # mention the brackets
    # todo check input

    starting_longitude = -73.555
    starting_latitude = 45.495
    goal_longitude = -73.585
    goal_latitude = 45.5275

    starting_node = get_graph_node(graph, starting_longitude, starting_latitude)
    goal_node = get_graph_node(graph, goal_longitude, goal_latitude)

    input("get heuristics")
    # get heuristic evaluations for each cell
    for node in flat_list:
        node.heuristic = heuristic(node, goal_node)

    input("get edges")
    # get specific edges
    get_edges(graph)

    if not starting_node.edges:
        message = "No moves available from starting point. Enter another location."
        #todo go back to main menu
    if not goal_node.edges:
        message = "No way to reach goal destination. Enter another destination."
        #todo go back main menu

    #todo
    # show graph, print stats
    print(starting_node.heuristic)

    # A* algorithm

    input("get a*")

    path = A_star_algorithm(starting_node, goal_node)
    if len(path) == 1:
        # todo output message
        # make sure a point is drawn on drawing + message !!!
        print("same")
        pass
    if not path:
        #output message no path found
        print("no solution")
        pass
    else:
        #draw path from path[]
        print(f"hurrah: {len(path)}")
        print(path)
        pass


# A point as represented in a 2D cartesian plane
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{{{self.x}, {self.y}}}"

    def __repr__(self):
        return self.__str__()


# represents a square area within the original map
# and contains crime statistics for the area
# Dimension is determined by GRID_SIDE_SIZE
# the boolean block means that the area will be represented as block
# due to a crime_count above the threshold
# It also contains a list of Edge objects representing edges to reachable nodes
# and the associated cost. This list will be useful for the implementation of the A* algorithm
class GraphNode:
    def __init__(self, point):
        self.point = point
        self.heuristic = -1    # heuristic value
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
class StateNode:
    def __init__(self, graph_node, parent, path_cost, A_star_score):
        self.graph_node = graph_node
        self.parent = parent
        self.children = []
        self.path_cost = path_cost
        self.A_star_score = A_star_score

    def __lt__(self, other):
        return self.A_star_score < other.A_star_score


# Building graph of points corresponding to bottom left corners of each grids
# Entering cumulative crime stats or each point
# Note that we use the conventional x, y cartesian representation to represent points
# whereas the geolocations in the shapefile use latitude/longitude (y,x)
def build_graph(shapes):
    y_axis_size = round((LATITUDE_RANGE[1] - LATITUDE_RANGE[0]) / GRID_SIZE)
    x_axis_size = round((LONGITUDE_RANGE[1] - LONGITUDE_RANGE[0]) / GRID_SIZE)

    grid = []

    # creating the basic graph nodes
    for row in range(y_axis_size):
        grid.append([])
        for col in range(x_axis_size):
            point = Point(LONGITUDE_RANGE[0] + GRID_SIZE * col, LATITUDE_RANGE[1] - GRID_SIZE * (row + 1))
            grid[row].append(GraphNode(point))

    # adding crime stats to appropriate grids
    for shape in shapes:
        longitude = shape.points[0][0]
        latitude = shape.points[0][1]
        cell = get_graph_node(grid, longitude, latitude)
        cell.crime_count += 1

    return grid


# gets a reference to the graph node representing a certain geolocation
def get_graph_node(grid, longitude, latitude):
    x = math.floor((longitude - LONGITUDE_RANGE[0]) / GRID_SIZE)
    y = math.floor((LATITUDE_RANGE[1] - latitude) / GRID_SIZE)
    return grid[y][x]


# returns distance between a node and the goal
# this heuristic is admissible and monotonous.
# It will never overestimate the true distance
# and, for cell x and neighbour x2, h(x) <= cost(x, x2) + h(x2)
# For these reasons, the goal will always be reached by the fastest route
# and **each node will always be first visited at the lowest cost**
def heuristic(node, goal_node):
    y = goal_node.point.y - node.point.y
    x = goal_node.point.x - node.point.x
    # the division by the sub-grid size is to put distances in the same units as the length cost
    return math.sqrt(y**2 + x**2) / GRID_SIZE


# Since the heuristic function is admissible, when we encounter the goal node, the solution path will
# also be the most effective path
# Note: since the heuristic function is monotonous, we store visited nodes in a set, since they will
# always be visited at the best cost the first time.
def A_star_algorithm(starting_node, goal_node, ):

    solution_path = []    # a list of points from the goal back to the root of the tree
    open_list = []        # priority queue, key is A* score, value is tree node reference
    closed_list = {None}  # this is a set because the heuristic function is monotonous

    root = StateNode(starting_node, None, 0, starting_node.heuristic)
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
            A_star_score = total_cost + edge.graph_node.heuristic
            tree_node = StateNode(edge.graph_node, node, total_cost, A_star_score)
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


# obtain edges for each cell in the grid
# Edges along the boundaries of the graph aren't allowed
def get_edges(grid):

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            node = grid[i][j]

            # East-bound edges
            if j < len(grid[0]) - 1:
                # if top-row
                if i == 0:
                    if node.block and grid[i+1][j].block:
                        pass  # no east edges
                    elif node.block:
                        node.edges.extend([Edge(grid[i][j+1], COST_ALONG_BLOCK),
                                          Edge(grid[i+1][j+1], COST_DIAGONAL)])
                    elif grid[i+1][j].block:
                        node.edges.append(Edge(grid[i][j+1], COST_ALONG_BLOCK))
                    else:
                        node.edges.extend([Edge(grid[i][j + 1], COST_STRAIGHT),
                                          Edge(grid[i + 1][j + 1], COST_DIAGONAL)])
                # if bottom-row
                elif i == len(grid) - 1:
                    if not node.block:
                        node.edges.append(Edge(grid[i-1][j+1], COST_DIAGONAL))
                # regular nodes
                else:
                    if node.block and grid[i+1][j].block:
                        pass  # no east edges
                    elif node.block:
                        node.edges.extend([Edge(grid[i][j+1], COST_ALONG_BLOCK),
                                          Edge(grid[i+1][j+1], COST_DIAGONAL)])
                    elif grid[i+1][j].block:
                        node.edges.extend([Edge(grid[i-1][j+1], COST_DIAGONAL),
                                           Edge(grid[i][j+1], COST_ALONG_BLOCK)])
                    else:
                        node.edges.extend([Edge(grid[i-1][j+1], COST_DIAGONAL),
                                           Edge(grid[i][j + 1], COST_STRAIGHT),
                                           Edge(grid[i + 1][j + 1], COST_DIAGONAL)])

            # West-bound edges
            if j > 0:
                # if top-row
                if i == 0:
                    if grid[i][j-1].block and grid[i+1][j-1].block:
                        pass  # no west edges
                    elif grid[i][j-1].block:
                        node.edges.extend([Edge(grid[i][j-1], COST_ALONG_BLOCK),
                                           Edge(grid[i + 1][j - 1], COST_DIAGONAL)])
                    elif grid[i + 1][j-1].block:
                        node.edges.append(Edge(grid[i][j - 1], COST_ALONG_BLOCK))
                    else:
                        node.edges.extend([Edge(grid[i][j - 1], COST_STRAIGHT),
                                           Edge(grid[i + 1][j - 1], COST_DIAGONAL)])
                # if bottom-row
                elif i == len(grid) - 1:
                    if not node.block:
                        node.edges.append(Edge(grid[i - 1][j - 1], COST_DIAGONAL))
                # regular nodes
                else:
                    if grid[i][j-1].block and grid[i + 1][j-1].block:
                        pass  # no east edges
                    elif grid[i][j-1].block:
                        node.edges.extend([Edge(grid[i][j - 1], COST_ALONG_BLOCK),
                                           Edge(grid[i + 1][j - 1], COST_DIAGONAL)])
                    elif grid[i + 1][j-1].block:
                        node.edges.extend([Edge(grid[i - 1][j - 1], COST_DIAGONAL),
                                           Edge(grid[i][j - 1], COST_ALONG_BLOCK)])
                    else:
                        node.edges.extend([Edge(grid[i - 1][j - 1], COST_DIAGONAL),
                                           Edge(grid[i][j - 1], COST_STRAIGHT),
                                           Edge(grid[i + 1][j - 1], COST_DIAGONAL)])

            # north-bound edge
            if i > 0 and j > 0:
                if node.block and grid[i][j-1].block:
                    pass
                elif node.block or grid[i][j-1].block:
                    node.edges.append(Edge(grid[i-1][j], COST_ALONG_BLOCK))
                else: # no blocks
                    node.edges.append(Edge(grid[i-1][j], COST_STRAIGHT))

            # south-bound edge
            if i < len(grid) - 1 and j > 0:
                if grid[i+1][j].block and grid[i+1][j-1].block:
                    pass
                elif grid[i+1][j].block or grid[i+1][j-1].block:
                    node.edges.append(Edge(grid[i+1][j], COST_ALONG_BLOCK))
                else: # no blocks
                    node.edges.append(Edge(grid[i+1][j], COST_STRAIGHT))


# graph_node class: has point, h(), crime_stat, edges

# tree_node class: has parent, graph_node, edges, A* score, cost, children


# edge class: has graph_node ref, cost.

# we could calculate h() on the fly just as edge

# open_list

# monotonous comment
# visited_list


# below shows range is ok

# for i_shape in shapes:
#     if i_shape.points[0][1] < 45.49 or i_shape.points[0][1] >= 45.53:
#         print(i_shape.points[0][1])
#         break
#
#
# print("hey")

# since monotonous, nodes are visited once, and thus edges are calculated once.
# Thus it makes sense to get the edges on the fly, vs calculateing all of them beforehand
# get_edges

if __name__ == "__main__":
    main()
