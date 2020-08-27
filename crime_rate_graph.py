import numpy as np
import math
import statistics as stats


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

    def __init__(self, shapes, resolution, threshold, lat_min, lat_max, long_min, long_max):

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

