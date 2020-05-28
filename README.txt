# COMP 472 - Assignment 1

## Required Libraries:

numpy
matplotlib
shapefile (PyShp)


## Usage:

All shapefiles must be located in a 'Shape' folder at the root of the project.

Upon running the main.py file, the user is granted with a menu.
The covered area, as well as the current resolution and threshold parameters are shown.
The user can choose to change the resolution (sub-grid size) as well as the threshold.

To begin, the user simply presses enter.
The user is shown the map, with its blocks shown in yellow,
as well as crime statistics (shown both on the map and on the terminal).
Each sub-area also shows its crime count.

The user is asked to supply starting and goal geo-locations.
The search algorithm is run and the solution path is then shown, as well as the length of the path.
on the map, and the specific coordinates are lso enumerated on the terminal screen.

The user presses enter to go back to the main menu to start over again or quit.
If no path was found, the user user is informed and the start and goal point are plotted
on the map toshow why no pat was found (one of the two point would be surrounded by blocks).

The user is prevented from entering out-of-scope geo-locations or parameters.

The resolution parameter is limited to the 0.0015 - 0.005 range.
Above 0.005 is simply not useful while below 0.0015 will not be practical on normally sized 
screens. However. The algorithm can perform at lower resolutions. It can be easily allowed.


## Information on the implementation:

### Crime Map

To store the crime map information, a graph, mapped to a 2D array is used. To run
the A* algorithm, a search tree is used.

The graph represents the crime map. Each node (GraphNode) represents a specific square-area the side length
of which is determined by the resolution attribute, passed as a parameter.
Each node has a crime count for its area, as well as tuple representing the geo-location of the
bottom-left point in the area. Real geo-locations are mapped to a node and take on the coordinates of
the sub-area it represents.

Nodes are mapped to a 2-dimensional array to facilitate the building of the graph and the edges. Each node
a set of edges. Edges contain a reference to a node, as well as a cost value.

Nodes also have a heuristic value, which is computed once the goal destination is known.


### A* algorithm

The A* algorithm is implemented with a state search tree.
A State Space Node contains a reference to a Graph Node (and all its relevant
information such as edges and heuristic values), as well as
information pertaining to the search algorithm.

It has a parent node (to trace back the solution path), and a set of child nodes
representing the nodes reachable from this node. It has a total path cost,
which is the cost from the start node to this node, according to the specific
branch in the search tree, as well as an A* score, which is the sum of the total path
cost and the related graph node's heuristic value.

The open list is stored in a min-heap queue (heapq).
The closed list is a set of tuples representing geo-locations.


### Details on the heuristic function:

The heuristic function returns the shortest absolute distance, in terms of resolution units (the sub-grid side size),
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
Instead, the heuristic will return the absolute minimum value in terms of resolution units (1 for straight
moves, 1.414 (or the square-root of 2) for immediate diagonals, and even more optimistic measures for farther
distances,because the measures allow to cut across segments).

MONOTONICITY
For the monotonous aspect, it follows from the previous description that,
for cell x1 and neighbour x2, h(x1) <= cost(x, x2) + h(x2), since no move to a reachable node
will ever improve the distance returned by the heuristic function, since it is already the shortest possible.

For these reasons, the goal will always be reached by the fastest route
and each node will always be visited at the lowest cost the first time they are encountered.

INFORMEDNESS
Some additional notes on informedness: the heuristic is informed, in the sense that it uses the overall distance
from the target. It is not too informed. It could have been more informed, but this would have been
costlier and error-prone: taking into account the presence of blocks would have make it easier to 
design a heuristic that is not admissible or monotonous, because of corner cases and other subtleties. Such a cost was simply not worth it. The focus was on designing an algorithm that works efficiently.
Furthermore, the A* algorithm is tasked with taking the blocks 
into account, so there is no need to try to do that work separately, on our own.



