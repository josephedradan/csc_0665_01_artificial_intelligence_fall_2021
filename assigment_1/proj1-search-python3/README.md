# Assignment 1 Solution

Question:
https://www.pooyanfazli.com/courses/csc665-fall21/assignments/assign1/project1.html

The Stuff run:

    # Introduction
        Files you'll edit:
            search.py	Where all of your search algorithms will reside.
            searchAgents.py	Where all of your search-based agents will reside.
            Files you might want to look at:
            pacman.py	The main file that runs Pacman games. This file describes a Pacman GameState type, which you use in this project.
            game.py	The logic behind how the Pacman world works. This file describes several supporting types like AgentState, Agent, Direction, and Grid.
            util.py	Useful data structures for implementing search algorithms.
            Supporting files you can ignore:
            graphicsDisplay.py	Graphics for Pacman
            graphicsUtils.py	Support for Pacman graphics
            textDisplay.py	ASCII graphics for Pacman
            ghostAgents.py	Agents to control ghosts
            keyboardAgents.py	Keyboard interfaces to control Pacman
            layout.py	Code for reading layout files and storing their contents
            autograder.py	Project autograder
            testParser.py	Parses autograder test and solution files
            testClasses.py	General autograding test classes
            test_cases/	Directory containing the test cases for each question
            searchTestClasses.py	Project 1 specific autograding test classes
        
    # Welcome to Pacman
        python pacman.py --layout testMaze --pacman GoWestAgent
        python pacman.py --layout tinyMaze --pacman GoWestAgent
        python pacman.py -h
    
    # Question 1 (3 points): Finding a Fixed Food Dot using Depth First Search
        python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
        python pacman.py -l tinyMaze -p SearchAgent
        python pacman.py -l mediumMaze -p SearchAgent
        python pacman.py -l bigMaze -z .5 -p SearchAgent
    
    # Question 2 (3 points): Breadth First Search
        python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
        python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
        python eightpuzzle.py

    # Question 3 (3 points): Varying the Cost Function
        python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
        python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
        python pacman.py -l mediumScaryMaze -p StayWestSearchAgent

    # Question 4 (3 points): A* search
        python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
    
    # Question 5 (3 points): Finding All the Corners
        Note: Make sure to complete Question 2 before working on Question 5, because Question 5 builds upon your answer for Question 2.
        python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
        python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem

    # Question 6 (3 points): Corners Problem: Heuristic
        python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
        Note: AStarCornersAgent is a shortcut for -p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic

    # Question 7 (4 points): Eating All The Dots
        python pacman.py -l testSearch -p AStarFoodSearchAgent
        Note: AStarFoodSearchAgent is a shortcut for -p SearchAgent -a fn=astar,prob=FoodSearchProblem,heuristic=foodHeuristic.
        Note: Make sure to complete Question 4 before working on Question 7, because Question 7 builds upon your answer for Question 4.
        python pacman.py -l trickySearch -p AStarFoodSearchAgent

    # Question 8 (3 points): Suboptimal Search
        python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5 

    # Object Glossary
        SearchProblem (search.py) A SearchProblem is an abstract object that represents the state space, successor function, costs, and goal state of a problem. You will interact with any SearchProblem only through the methods defined at the top of search.py
        
        PositionSearchProblem (searchAgents.py) A specific type of SearchProblem that you will be working with --- it corresponds to searching for a single pellet in a maze.
        
        CornersProblem (searchAgents.py) A specific type of SearchProblem that you will define --- it corresponds to searching for a path through all four corners of a maze.
        
        FoodSearchProblem (searchAgents.py) A specific type of SearchProblem that you will be working with --- it corresponds to searching for a way to eat all the pellets in a maze.
        
        Search Function is a function which takes an instance of SearchProblem as a parameter, runs some algorithm, and returns a sequence of actions that lead to a goal. Example of search functions are depthFirstSearch and breadthFirstSearch, which you have to write. You are provided tinyMazeSearch which is a very bad search function that only works correctly on tinyMaze
        
        SearchAgent is a class which implements an Agent (an object that interacts with the world) and does its planning through a search function. The SearchAgent first uses the search function provided to make a plan of actions to take to reach the goal state, and then executes the actions one at a time.







