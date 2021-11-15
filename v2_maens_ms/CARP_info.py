import numpy as np

get_str = lambda s: s.split(': ')[-1].rstrip('\n')
get_int = lambda s: int(get_str(s))


class Edge:
    def __init__(self, u, v, cost, demand):
        self.u = u
        self.v = v
        self.cost = cost
        self.demand = demand

    def __hash__(self):
        return hash((self.u, self.v)) + hash((self.v, self.u))

    def __eq__(self, other):
        return self.u, self.v == other.u, other.v or self.u, self.v == other.v, other.u

    def __str__(self):
        return str((self.u, self.v, self.cost, self.demand))


def floyd(matrix):
    """
    Notes:
    * If there is no edge connecting i->j then matrix[i,j] should be numpy.inf.
    * The mains diagonal of matrix is zero.

    :param matrix: An NxN NumPy array.
                matrix[i,j] = edge cost from node i to node j
    :return: An NxN NumPy array s.t. result[i,j] is the shortest distance to travel from node i to node j.
            If no such path exists then result[i,j] is numpy.inf
    """
    n = len(matrix)
    for i in range(n):
        matrix = np.minimum(matrix, matrix[np.newaxis, i, :] + matrix[:, i, np.newaxis])
    return matrix


class Info:
    # Information of carp instance
    def __init__(self, filename):
        self.instance = open(filename, 'r', 51200).readlines()  # 50 KB buffer

        info = self.instance[:8]
        self.name = get_str(info[0])
        self.vertices = get_int(info[1])
        self.depot = get_int(info[2])
        self.edges_required = get_int(info[3])
        self.edges_non_req = get_int(info[4])
        self.vehicles = get_int(info[5])
        self.capacity = get_int(info[6])
        self.total_cost = get_int(info[7])

        self.edges = dict()  # all edges
        self.tasks = dict()  # all required edges
        data = map(lambda s: s.split(), self.instance[9:-1])
        # full of inf, ps: the vertices +1 means start from 1
        matrix = np.full((self.vertices + 1, self.vertices + 1), np.inf)
        np.fill_diagonal(matrix, 0)  # make diagonal 0

        for line in data:
            u, v, cost, demand = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            edge = Edge(u, v, cost, demand)
            self.edges[(u, v)] = edge
            self.edges[(v, u)] = edge
            if demand:  # demand != 0
                self.tasks[(u, v)] = edge
                self.tasks[(v, u)] = edge
            matrix[u, v] = cost
            matrix[v, u] = cost
        self.min_dist = floyd(matrix)
