# Artificial Intelligence Assignment  
Author: Simon Brillant-Giroux  
  
## General Description  
  
Implementation of a A* search algorithm to find the the shortest route between two coordinates across a map, given certain limitations based on crime-rate data.  
  
  
## Required Libraries:  
  
 - numpy 
 - matplotlib 
 - shapefile (PyShp)  
  
## Usage:  
  
 All shapefiles must be located in a 'Shape' folder at the root of the project. Upon running the path_finder.py file, the main menu is displayed. The current settings are displayed as well as an options menu.  
  
The settings consist of the covered area, as well as the current resolution and threshold parameter. The threshold determines which nodes are going to be blocked as a function of their crime rate and is specified as a percentage of nodes to be accessible in the graph. The covered area is hard-coded for this assignment but can be changed at the top of path_finder.py.
  
The options menu lets the user change the resolution (grid node size), change the threshold, or start with the current options.  

When the user chooses to start with the current options, the resulting map is shown, with its blocks shown in yellow, as well as crime statistics (shown both on the map and on the terminal). Each sub-area also shows its crime count. The user can choose to return to the main menu to reconfigure the map, or continue with the current map. 

The user is then prompted to supply starting and destination geolocations. The search algorithm is run and the solution path is then graphically shown on the map, as well as the length of the path.The specific coordinates are also enumerated on the terminal screen. 

The user presses enter to go back to the main menu to start over again or quit.  
  
If no path was found, the user user is informed of that and the start and goal point are plotted on the map to show why no path was found (one of the two point would be surrounded by blocks, or the two points are isolated by blocks).  
  
 The user is prevented from entering out-of-scope geolocations or parameters. The resolution parameter is limited to the 0.001 - 0.005 range. Above 0.005 is simply not useful while below 0.001 will not be practical on normally sized screens. 
  
## Information on the implementation:  
  
### Crime Map  
  
To store the crime map information, a graph, mapped to a 2D array is used. To run the A* algorithm, a search tree is used.  

The graph represents the crime map. Each node (GraphNode) represents a specific square-area the side length of which is determined by the resolution attribute, passed as a parameter. Each node has a crime count for its area, as well as tuple representing the geolocation of the bottom-left point in the area. Real geolocations are mapped to a node and take on the coordinates of the sub-area it represents.  

Nodes are mapped to a 2-dimensional array to facilitate the building of the graph and the edges. Each node has a set of edges. Edges contain a reference to a node, as well as a cost value. The cost value is supplied in the assignment: 
- straight edge: 1
- diagonal edge: 1.5
- straight edge along a block: 1.3   

Nodes also have a heuristic value, which is computed once the goal destination is known.  
  
### A* algorithm  
  
The A* algorithm is implemented with a state search tree made of state space nodes. 

A StateSpaceNode contains a reference to a GraphNode (and all its relevant information such as edges and heuristic values), as well as information pertaining to the search algorithm. It has a parent node (to trace back the solution path), and a set of child nodes representing the reachable nodes. It has a total path cost, which is the cost from the start node to this node, according to the specific branch in the search tree, as well as an A* score, which is the sum of the total path cost and the related graph node's heuristic value.  

The open list is stored in a min-heap queue (heapq). The closed list is a set of tuples representing geo-locations.  
  
### Details on the heuristic function:  
  
The heuristic function returns the euclidean distance between the node and the goal, in terms of resolution units from the original crime map, between a node and the goal. It measures the distance between a point and the goal using the Pythagorean theorem, and then divides the result by the resolution (or the size of the side of the square area represented by each node) to put the result in the same units as the path cost.
  
This heuristic is admissible and monotonous.  

#### Admissibility

The heuristic is admissible because it calculates the absolute shortest distance between a point and the goal. Therefore, it will never overestimate the true cost of a path. The real cost is limited to straight and diagonal moves from one node to the other, ascribing 1 or 1.3 units to straight moves and 1.5 to diagonal moves, and cannot go through blocks. Instead, the heuristic will return the absolute minimum value in terms of resolution units (1 for straight moves, 1.414 (or the square-root of 2) for immediate diagonals, and even more optimistic measures for farther distances,because the measures allow to cut across segments).
  
#### Monotonicity 

For the monotonous aspect, it follows from the previous description that, for cell $x1$ and neighbor $x2$, $h(x1) <= cost(x, x2) + h(x2)$, since no move to a reachable node will ever improve the distance returned by the heuristic function, since it is already the shortest possible.

#### Implications of the Heuristic
For these reasons, the fastest route to the goal will always be returned and each node will always be visited at the lowest cost the first time they are encountered.  
 
#### Informedness
Some additional notes on informedness: the heuristic is informed, in the sense that it uses the overall distance from the target. It could have been more informed, but this would have been costlier and error-prone: taking into account the presence of blocks would have make it easier to  design a heuristic that is not admissible or monotonous, because of corner cases and other subtleties. Such a cost was simply not worth it. The focus was on designing an algorithm that works efficiently.  

Furthermore, the A* algorithm is tasked with taking the blocks into account, so there is no need to try to do that work separately, on our own.

