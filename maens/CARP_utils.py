import numpy as np


class Solution:
    def __init__(self, task_seq, loads, quality, exceed_load):
        self.task_seq = task_seq
        self.loads = loads
        self.quality = quality
        self.exceed_load = exceed_load
        self.fitness = 0

    def calculate_fitness(self, coef):
        self.fitness = self.quality + coef * self.exceed_load

    def __hash__(self):
        return hash((self.quality, self.exceed_load))

    def __eq__(self, other):
        return self.quality == other.quality and self.exceed_load == other.exceed_load


class Insert:
    def __init__(self, task, routeID, position, cost, exceed_load):
        self.task = task
        self.routeID = routeID
        self.position = position
        self.cost = cost
        self.exceed_load = exceed_load


class Move:
    def __init__(self, type, task1=0, task2=0,
                 orig_seg=0, targ_seg=0, orig_pos=0, targ_pos=0,
                 quality=np.inf, exceed_load=np.inf):
        # type = 1, 2, 3 -> single insert, double insert, swap
        self.type = type
        self.task1 = task1
        self.task2 = task2
        self.orig_seg = orig_seg
        self.targ_seg = targ_seg
        self.orig_pos = orig_pos
        self.targ_pos = targ_pos
        self.quality = quality
        self.exceed_load = exceed_load
        self.fitness = np.inf

    def calculate_fitness(self, coef):
        self.fitness = self.quality + coef * self.exceed_load



