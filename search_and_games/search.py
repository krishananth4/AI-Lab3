# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "Up", 4)
    >>> B1 = Node("B", S, "Down", 3)
    >>> B2 = Node("B", A1, "Left", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"
    # Initialize a queue for BFS traversal
    frontier = util.Queue()
    # Initialize a set to keep track of visited nodes
    visited = set()
    # Initialize the starting node
    start_node = problem.getStartState()
    # Enqueue the starting node into the frontier queue
    frontier.push((start_node, []))

    # Continue BFS traversal until the frontier queue is empty
    while not frontier.isEmpty():
        # Dequeue a node from the frontier queue
        current_node, actions = frontier.pop()

        # Check if the current node is the goal state
        if problem.goalTest(current_node):
            # If it is, return the list of actions to reach the goal
            return actions

        # If the current node has not been visited yet
        if current_node not in visited:
            # Mark the current node as visited
            visited.add(current_node)

            # Expand the current node by generating its successors
            for action in problem.getActions(current_node):
                # Get the successor state resulting from applying the action
                successor = problem.getResult(current_node, action)
                # Append the action to the list of actions taken so far
                next_actions = actions + [action]
                # Enqueue the successor node into the frontier queue
                frontier.push((successor, next_actions))

    # If no solution is found, return an empty list
    return []
    # util.raiseNotDefined()
    
def depthFirstSearch(problem): 

    "*** YOUR CODE HERE ***"   
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)

    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
def UniformCostSearch(problem):
    """Search the node that has the lowest path cost first."""
    "*** YOUR CODE HERE ***"  
    # util.raiseNotDefined()
    

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Import searchAgents.py to include manhattanHeuristic method 
    import searchAgents
    heuristic = searchAgents.manhattanHeuristic
    # Initialize a list of actions that provides information on the steps it took to get to a node
    actionsList = []
    # Here a set is initialized to keep track of the nodes that have been visited
    visited = set()
    # The frontier is implemented as a Priority Queue where the class is defined in util.py
    frontier = util.PriorityQueue()
    # The frontier set is used to keep track of the nodes in the frontier Priority Queue
    frontierSet = set()
    # The start node is initialized and calls the start state method to get the start node
    startNode = problem.getStartState()
    # The g Function is initalized to 0, which will be used in calculation of the total cost
    gFunction = 0
    # The h function is initalized at the start state where heuristic() calculates the heurstic value of the start state. This uses the manhattanHeuristic.
    hFunction = heuristic(startNode, problem)
    # The start node and current list of actions tuple is pushed onto the frontier along with the priority stated by the heuristic cost. This is first node that is kept track of in the frontier.
    frontier.push((startNode, actionsList), hFunction)
    # Simultaneously, the start ndoe is added to the frontier set
    frontierSet.add(startNode)

    # If the frontier is empty then we return an empty list because there is no further actions to take
    if frontier.isEmpty():
        return []

    # Until the frontier is not empty
    while not frontier.isEmpty():

        # The current visited node is popped from the frontier and the actions are taken accounted for
        currentVisitingNode, actions = frontier.pop()

        # If the current visited node is a goal node then we return the list of actions to get to that node
        if problem.goalTest(currentVisitingNode):
            return actions
        
        # If the current visiting node has not been visited in previous iterations, then we add this node to the visited set
        if currentVisitingNode not in visited:
            visited.add(currentVisitingNode)

        # Here we loop through to see the actions we can take from the current visiting node
        for action in problem.getActions(currentVisitingNode):
            # The action we can take from the current visiting node results in getting the successor (which is a child to the current visiting node)
            successor = problem.getResult(currentVisitingNode, action)
            # Add the action that was just taken onto the list of actions, which continues to keep track of all the actions we have taken so far
            actionsList = actions + [action]
            # The g function is calculated to keep track of the path cost from the start node to the successor (getCostOfActions() helper function is used to simplify this process)
            gFunction = problem.getCostOfActions(actionsList)
            # The h function calculates the heuristic value of the successor to the goal node, which is the estimated cheapest cost to get from a a given node to the goal node. This is using the manhattanHeuristic.
            hFunction = heuristic(successor, problem)
            # The total cost (f(n)) is then calcuated by adding the heursitic value and the path cost from the start node 
            totalCost = gFunction + hFunction
            
            # Here we do a check to ensure the successor is not in the frontier and not in the visited set because we don't want to add an already visited node to the frontier
            if successor not in visited and successor not in frontierSet:
                # Then we add the successor onto the frontier priority queue with the priority as the total cost of the actions
                frontier.push((successor, actionsList), totalCost)
                # Simultaneously, we also add the successor onto the frontier set using the add() method
                frontierSet.add(successor)
    
    # Otherwise, if we have no goal node, then return an empty list symbolizing that no actions have been taken
    return []

    # util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
