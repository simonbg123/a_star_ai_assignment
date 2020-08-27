import heapq


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


def a_star_algorithm(starting_node, goal_node):
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

    # early return when start or goal is surrounded by blocks
    if len(goal_node.edges) < 1 or len(starting_node.edges) < 1:
        return solution_path

    root = StateSpaceNode(starting_node, None, 0, starting_node.heuristic)
    heapq.heappush(open_list, root)

    while open_list:

        # # assignment requirement: checking timing requirement
        # if time.time() - start_time > 10.0:
        #     print("Time is up. The optimal path is not found.")
        #     input("Press enter to terminate program.")
        #     exit()

        node = heapq.heappop(open_list)

        if node.graph_node is goal_node:
            # we have reached the goal
            solution_path = get_solution_path(node)
            return solution_path

        closed_list.add(node.graph_node.point)

        for edge in node.graph_node.edges:
            if edge.graph_node.point in closed_list:
                # ignore a visited node. Because of the monotonicity of the heuristic (see Graph),
                # we don't need to compare the A* scores: a node is always visited at its lowest possible score
                continue
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
