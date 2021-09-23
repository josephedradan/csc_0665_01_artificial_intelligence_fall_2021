Joseph Edradan
9/23/2021
920783419

In search.py (Questions 1 to 4), all the searching algorithms (DFS, BFS, UCS, A*) use the same base algorithm that
contains a data structure and a loop to pop from that data structure to then do stuff with. The name of
that function is called generic_search_algorithm_base. I made a class called Container rather than use the
tuple trippe for cleaner use of accessing data about int the tuple triple. It also knows what the previous
Container it was connected to was which allows a path to be created.

In searchAgents.py (Question 5), the CornersProblems uses a custom class called HashableGoal that is firstly hashable
and mainly used to transfer information about which position on the grid Pacman as traveled to. HashableGoal should know
about which location and what order a goal has been reached by the Pacman, this contributes to the hash of the
HashableGoal object itself. The reason why the the order and position of a goal is used in the hash is because it
is used to differentiate positions that the Pacman has went to in the algorithms in search.py's visited set.

For cornersHeuristic (Question 6), I have 7 solutions, but 2 of them work, the one that I uncommented out is my V4 solution
which is to make all permutations of all the corners and then run the Manhattan Distance on the permutation of corners
sequentially starting from Pacman's current position. Then select the smallest distance from the sum of the Manhattan
Distances of each permutation and then return that smallest distance as the heuristic. V5 solution works too, but
it creates the distance by selecting the smallest distance when connecting to a new position sequentially and repeats
that process until there are no more positions to go to which results into a sum of distances that is then returned.

For eating All the Dots (Question 7), I have 4 solutions, and 2 of them work. I uncommented out V4 because this problem
is basically run a bfs to each Food position and then return the shortest distance one. However, returning the shortest
distance is bad because it will expand too much so you need to use max instead of min. The explanation for using max
over min is in the comments of the code, but i'll post it here too:

        "Using the max distance instead of min distance is essentially equivalent to returning the longest distance
        for the priority queue algorithm to then select the best of the (Big Heuristic Cost + Node Cost distance).
        Returning the longest distance (Big Heuristic Cost) is like saying that the path you will
        make to that node is the worst. For the Priority queue, it will select the shortest of the longest
        distance sums (Node Cost + Heuristic Cost). The series of big Heuristic Costs in the PQ will give you a
        solution that expands the least amount of nodes. Longer distances are more influential than short
        distances meaning Heuristic Cost > Node Cost most of the time so the PQ will select mainly based
        on Heuristic Cost.

        If you were to use min then you would select the shortest distance to a node and so
        Heuristic Cost < Node Cost most of the time for the priority queue. So the PQ would be mostly
        selecting based on Node Cost which is close to/is a pure BFS or UCS depending on implementation."

The V3 solution is the alternative solution to V4 which is pretty much the same algorithm as V4, but uses a modified
version of V4 of problem 6 (cornersHeuristic) as a base to solve the problem. The primary reason I use Solution V4 is
because mazeDistance is exactly what is needed for this problem and you can cache the inputs of that function too
so the runtime of the autograder.py will be low.

Also, for question 7, if you understood what foodGrid.asList() did and is, it would be very helpful for Question 8.
Plus, the idea of foodGrid.asList() is exactly what I used in cornersHeuristic (Question 6), because
foodGrid.asList() returns an iterable of the remaining positions of food or goals.

For Suboptimal Search (Question 8), it's basically question 7, but you have access to "problem" and you want a path
which is what the algorithms in search.py do, so just return the result of the a star algorithm and you get your
answer.
