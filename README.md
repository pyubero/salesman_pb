# The travelling salesman in the real world

In this repo I use python and OpenStreetMaps to demonstrate the solution of the travelling salesman problem in a real world scenario. The general idea of the problem is the following:

>Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?`

And although it typically involves travelling to different cities, I just consider about 20 different coordinates in the same city (Madrid in this case).

### The greedy algorithm provides an initial route
The greedy solution selects the next target coordinate as the one that is closest to the current position and starting at the green dot. By using a greedy algorithm the solution is quite satisfactory in comparison with a randomly chosen one (see plot).

### Simulated annealing optimizes the route
The greedy solution is then fed to a simulated annealing (SA) algorithm to optimize the route. SA tries to find better solutions (local minima of distrance travelled) by allowing suboptimal changes in the route, this depends on the temperature parameter. At every timestep, the algorithm becomes slightly more strict, and by the end (when the temperature is ~0) it becomes simply the gradient descent. This step allows to "untangle" some knots in the route.

## Example in Madrid
Here I show an example in the city of Madrid where a salesman needs to go to 20 different locations and return to the initial starting point. Although the greedy algorithm provides a solution with just 78km (about half of a random route), the SA optimization steps minimizes this distance to just 66km. The starting and ending point is the green circle, the target locations are the red circles, and the route is shown in red. 

<img src="https://github.com/pyubero/salesman_pb/blob/main/figure_map.png" width="600">

<img src="https://github.com/pyubero/salesman_pb/blob/main/graph.png" width="400">
