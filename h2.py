# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions 
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import pickle
import math
from copy import deepcopy
import os


class PriorityQueue(object):
    """A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue.
        """
        self.queue = []
        heapq.heapify(self.queue)
    def pop(self):
        """Pop top priority node from queue.
        Returns:
            The node with the highest priority.
        """
        return heapq.heappop(self.queue)

    def remove(self, node_id):
        """Remove a node from the queue.
        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.
        Args:
            node_id (int): Index of node in queue.
        """
        self.queue.remove(node_id)
        heapq.heapify(self.queue)
    
    def __iter__(self):
        """Queue iterator.
        """
        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queuer to string.
        """
        return 'PQ:%s' % self.queue

    def append(self, node):
        """Append a node to the queue.
        Args:
            node: Comparable Object to be added to the priority queue.
        """
        heapq.heappush(self.queue, node)

    def getValue(self, key):
        for pair in self.queue:
            if pair[1][-1] == key:
                return pair
        return None

    def __contains__(self, key):
        """Containment Check operator for 'in'
        Args:
            key: The key to check for in the queue.
        Returns:
            True if key is found in queue, False otherwise.
        """
        return key in [n[-1] for _, n in self.queue]

    def __eq__(self, other):
        """Compare this Priority Queue with another Priority Queue.
        Args:
            other (PriorityQueue): Priority Queue to compare against.
        Returns:
            True if the two priority queues are equivalent.
        """
        return self == other

    def size(self):
        """Get the current size of the queue.
        Returns:
            Integer of number of items in queue.
        """
        return len(self.queue)
    
    def clear(self):
        """Reset queue to empty (no nodes).
        """
        self.queue = []
        
    def top(self):
        """Get the top item in the queue.
        Returns:
            The first item stored in teh queue.
        """
        return self.queue[0]

def breadth_first_search(graph, start, goal):
    """Warm-up exercise: Implement breadth-first-search.
    See README.md for exercise description.
    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = []
    frontier.append([start])   
    visited = set()
    while frontier:
        #print frontier
        path = frontier.pop(0)
        cur_node = path[-1]
        if cur_node in visited:
            continue
        else:
            visited.add(cur_node)
            if cur_node == goal:
                return path
            for neighbor in graph[cur_node]:
                #print "Neighbor: " + str(path) + " " + str(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                frontier.append(new_path)

    return []

def uniform_cost_search(graph, start, goal):
    """Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    frontier = PriorityQueue()
    frontier.append((1,[start]))
    visited = set()
    if start == goal:
        return []
    while frontier:
        tmp_node = frontier.pop()
        cur_length = tmp_node[0]
        path = tmp_node[-1]
        cur_node = path[-1]
        if cur_node in visited:
            continue
        else:
            visited.add(cur_node)
            if cur_node == goal:
                return path
            for neighbor, weights in graph[cur_node].iteritems():
                weight = weights['weight'] + cur_length
                if(neighbor not in visited):
                    new_path = list(path)
                    new_path.append(neighbor)
                    frontier.append((weight,new_path))
    return []


def null_heuristic(graph, v, goal ):
    """Null heuristic used as a base line.

    Args:
        graph (explorable_graph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node as a list.
    """
    x1, y1 = graph.node[v]['pos']
    x2, y2 = graph.node[goal]['pos']

    return math.hypot(x2 - x1, y2 - y1)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """ Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    frontier = PriorityQueue()
    frontier.append((1,[0,start]))
    visited = set()

    #print "MIN DISTANCE:" + str(heuristic(graph,start,goal))
    if start == goal:
        return []
    while frontier:
        #print "---------------"
        tmp_node = frontier.pop()
        cur_length = tmp_node[0]
        path = tmp_node[-1]
        dist_traveled = path.pop(0)
        cur_node = path[-1]
        #print cur_length
        #print cur_node
        #print dist_traveled
        if cur_node in visited:
            #print "Already Visited"
            continue
        else:
            if cur_node == goal:
                return path
            visited.add(cur_node)
            for neighbor, weights in graph[cur_node].iteritems():
                heur = heuristic(graph,neighbor,goal)
                weight = weights['weight']
                #print neighbor + " " + str(weight) + " " + str(heur) + " " + str(dist_traveled)
                distance = weight + heur + dist_traveled
                new_path = [weight + dist_traveled] + list(path)
                new_path.append(neighbor)
                if neighbor not in visited and neighbor not in frontier:
                    frontier.append((distance, new_path))
                elif neighbor in frontier:
                    old_dist = frontier.getValue(neighbor)[0]
                    if(distance < old_dist):
                        frontier.append((distance, new_path))
            #print "Frontier:"
            #print frontier
            #print "Visited:"
            #print visited

    return []


def bidirectional_ucs(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier.append((0,[start]))
    reverse_frontier = PriorityQueue()
    reverse_frontier.append((0,[goal]))
    visited = set()
    back_visited = set()
    #print "Frontier:"
    #print frontier
    #print "Reverse Frontier:"
    #print reverse_frontier
    isFront = True
    #print "Start: " + str(start)
    #print "Goal: " + str(goal)
    while frontier and reverse_frontier:
        #print "------------------"
        #print "isFront = " + str(isFront)
        tmp_node = None
        if isFront:
            tmp_node = frontier.pop()
            #print "Next Reverse: " + str(reverse_frontier.top()[-1][-1])
        else:
            tmp_node = reverse_frontier.pop()
        cur_length = tmp_node[0]
        path = tmp_node[-1]
        cur_node = path[-1]
        #print "Current Node: " + str(cur_node)
        if isFront:
            if cur_node in visited:
                #print "Already Visited"
                continue
            else:
                visited.add(cur_node)
        elif not isFront:
            if cur_node in back_visited:
                #print "Already Visited"
                continue
            else:
                back_visited.add(cur_node)
          
        for neighbor, weights in graph[cur_node].iteritems():
            #print "Neighbors"
            #print neighbor
            weight = weights['weight'] + cur_length
            if isFront:
                if neighbor in back_visited:
                    del path[-1]
                    full_path = path + list(reversed(reverse_frontier.getValue(cur_node)[-1]))
                    #print "Found Value: " + str(full_path)
                    return full_path
                    break
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    frontier.append((weight,new_path))
            else:
                if neighbor in visited:
                    del path[-1]
                    full_path = frontier.getValue(cur_node)[-1] + list(reversed(path))
                    #print "Found Value: " + str(full_path)
                    return full_path
                    break
                if neighbor not in back_visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    reverse_frontier.append((weight,new_path))

        isFront = not isFront
        #print "Frontier:"
        #print frontier
        #print "Reverse Frontier:"
        #print reverse_frontier
    return []


def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = PriorityQueue()
    explored = []
    frontier.append(start)

    while frontier:
        cur_node = frontier.pop()
        explored.append(cur_node)

        for neighbor in graph[cur_node]:
            if neighbor == goal:
                explored.append(neighbor)
                return explored
            if neighbor not in explored:
                frontier.append(neighbor)

    return []


# Extra Credit: Your best search method for the race
#
def load_data():
    """Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    pickle_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data


def custom_search(graph, start, goal, data=None):
    """Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    frontier = PriorityQueue()
    explored = []
    frontier.append(start)

    while frontier:
        cur_node = frontier.pop()
        explored.append(cur_node)

        for neighbor in graph[cur_node]:
            if neighbor == goal:
                explored.append(neighbor)
                return explored
            if neighbor not in explored:
                frontier.append(neighbor)

    return []
