import numpy as np


class Solution:
    def __init__(self, routes, loads, costs, total_cost, capacity):
        self.routes = routes
        self.loads = loads
        self.costs = costs
        self.total_cost = int(total_cost) if total_cost != np.inf else np.inf
        self.capacity = capacity

        self.invalid_generations = 0
        self.is_valid = True
        self.discard_chance = 0
        if self.loads:
            self.validate()

    def validate(self):
        exceed = sum([c - self.capacity for c in self.loads if c > self.capacity])
        if exceed != 0:
            self.invalid_generations += 1
            self.is_valid = False
        else:
            self.is_valid = True
        self.discard_chance = 2 * exceed / sum(self.loads) * pow(3, self.invalid_generations)

        for i, item in enumerate(self.routes):
            if not item:
                del self.routes[i]
                del self.loads[i]
                del self.costs[i]


    def quality(self, info):
        route_cost = self.route_cost(info)
        return sum(route_cost)

    def route_cost(self, info):
        routes = self.routes
        route_cost = []
        for k, route in enumerate(routes):
            curr = info.min_dist[info.depot, route[0][0]]
            for i in range(len(route)):
                u, v = route[i]
                next_u = route[i + 1][0] if i != len(route) - 1 else info.depot
                curr += info.edges[(u, v)].cost + info.min_dist[v, next_u]
            route_cost.append(curr)
        return route_cost


    def generate_new(self, info):
        new_solution = Solution(self.routes, [], [], 0, self.capacity)
        costs = new_solution.route_cost(info)
        loads = []
        for i, route in enumerate(self.routes):
            loads.append(sum(map(lambda r: info.tasks[(r[0], r[1])].demand, route)))
        new_solution.loads = loads
        new_solution.costs = costs
        new_solution.total_cost = sum(costs)
        new_solution.validate()
        return new_solution

    def __hash__(self):
        return hash(str(self.routes))

    def __eq__(self, other):
        return self.routes == other.routes

    def __str__(self):
        return '\n'.join(['routs:' + str(self.routes), 'loads:' + str(self.loads), 'costs:' + str(self.costs), 'total_cost:' + str(self.total_cost),
                          'is_valid:' + str(self.is_valid), 'non_valid_generations:' + str(self.invalid_generations)])