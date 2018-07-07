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
import pdb


class Node(object):
    """
    This is a helpful self-written class for solving search problem.
    """

    def __init__(self, _state, path, path_cost, action=None):
        self.state = _state
        self.path = path
        self.path_cost = path_cost
        self.action = action


    def __repr__(self):
        return str({'state:': self.state, 'path:': self.path, ' cost:': self.path_cost, 'action:': self.action})

    def __str__(self):
        """
        Helpful for checking problem.
        """

        return str({'state:': self.state, 'path:': self.path, ' cost:': self.path_cost, 'action:': self.action})


def solution(node):
    return node.path


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

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())

    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    node = Node(problem.getStartState(), path=[], path_cost=0)
    _explored = set()
    frontier = util.Stack()
    frontier.push(node)
    # When luckily the first state is goal:
    if problem.isGoalState(state):
        return solution(node)

    while not frontier.isEmpty():
        node = frontier.pop()
        _explored.add(node.state)
        for successor in problem.getSuccessors(node.state):
            state, action, _ = successor
            # print action
            path = node.path + [successor[1]]
            child = Node(successor[0], path=path, path_cost=0)

            if child.state not in _explored:
                if problem.isGoalState(child.state):
                    return solution(child)
                frontier.push(child)
    raise 'This maze has no solution.'


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # state = problem.getStartState()
    # node = Node(problem.getStartState(), path=[], path_cost=0)
    # _explored = set()
    # frontier = util.Queue()
    # frontier.push(node)
    # # When luckily the first state is goal:
    # if problem.isGoalState(state):
    #     return solution(node)
    #
    # while not frontier.isEmpty():
    #     node = frontier.pop()
    #     print node.state
    #     _explored.add(node.state)
    #     for successor in problem.getSuccessors(node.state):
    #         state, action, _ = successor
    #         # print action
    #         path = node.path + [successor[1]]
    #         cost = node.path_cost+1
    #         child = Node(successor[0], path=path, path_cost=cost)
    #
    #         if child.state not in _explored:
    #             if problem.isGoalState(child.state):
    #                 return solution(child)
    #             frontier.push(child)
    # raise 'This maze has no solution.'
    state = problem.getStartState()
    print(state)
    node = Node(state,[],0)
    if problem.isGoalState(state):
        return solution(node)
    frontier = util.Queue()
    frontier.push(node)
    explored = set()
    explored.add(node.state)
    while True:
        if frontier.isEmpty():
            raise 'This maze has no solution'
        node = frontier.pop()
        explored.add(node.state)
        for successor in problem.getSuccessors(node.state):
            path = node.path + [successor[1]]
            # print path
            cost = node.path_cost+1
            child = Node(successor[0],path,cost)

            if child.state not in explored:
                if child.state not in frontier.list:
                    if problem.isGoalState(child.state):
                        return solution(child)
                    frontier.push(child)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    node = Node(problem.getStartState(), path=[], path_cost=0)
    _explored = set()
    frontier = util.PriorityQueue()
    frontier.push(node, node.path_cost)
    # When luckily the first state is goal:
    if problem.isGoalState(node.state):
        return solution(node)

    while not frontier.isEmpty():
        node = frontier.pop()
        _explored.add(node.state)
        for successor in problem.getSuccessors(node.state):

            path = node.path + [successor[1]]
            child = Node(successor[0], path=path, path_cost=node.path_cost + successor[2])
            if child.state not in _explored:
                if problem.isGoalState(child.state):
                    return solution(child)
                frontier.update(child, child.path_cost)

    raise 'This maze has no solution.'
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    "*** YOUR CODE HERE ***"
    node = Node(problem.getStartState(), path=[], path_cost=0)
    _explored = set()
    frontier = util.PriorityQueue()
    frontier.push(node, node.path_cost)
    # When luckily the first state is goal:
    if problem.isGoalState(node.state):
        return solution(node)

    while not frontier.isEmpty():
        node = frontier.pop()
        _explored.add(node.state)
        for successor in problem.getSuccessors(node.state):
            path = node.path + [successor[1]]
            child = Node(successor[0], path=path, path_cost=node.path_cost + successor[2])
            if child.state not in _explored:
                if problem.isGoalState(child.state):
                    return solution(child)
                frontier.update(child, child.path_cost + heuristic(child.state, problem))
    raise 'This maze has no solution.'


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
