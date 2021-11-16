import copy
import numpy as np

from v2_maens_ms.CARP_info import Edge


class Node:
    """A node in G* of ulusoy split"""

    def __init__(self, u, v, cost):
        self.u = u
        self.v = v
        self.cost = cost

    def __hash__(self):
        return hash((self.u, self.v)) + hash((self.v, self.u))

    def __eq__(self, other):
        return self.u, self.v == other.u, other.v or self.u, self.v == other.v, other.u

    def __str__(self):
        return str((self.u, self.v, self.cost))


class Ulusoy:
    # ulusoy's splitting implementation
    def __init__(self, info):
        self.info = info
        self.dummy_edge = Edge(info.depot, info.depot, 0)

    def flatten(self, tasks):
        dummy_edge = self.dummy_edge
        # for route in tasks:
        #     route.append(dummy_edge)
        tasks[0].insert(0, dummy_edge)
        tasks[-1].append(dummy_edge)
        # flatten
        tasks = sum(tasks, [])

    def partition(self, tasks):
        info = self.info
        size = len(tasks)
        graph = np.full((size, size), np.inf)
        np.fill_diagonal(graph, 0)

        for i in range(1, size-1):
            graph[0, i] = info.min_dist[info.depot, tasks[i].u] \
                          + tasks[i].cost \
                          + info.min_dist[tasks[i].v, info.depot]

        for i, task in enumerate(tasks):
            """ C2 
                Ck,h is the sum of the node costs Ci,j,included in
                the vehicle tour and the fixed cost of the least
                capacity vehicle meeting the demand of the corresponding vehicle tour
            """
            load = task.load
            cost = info.dist[info.depot, task.u] + task.cost
            j = i + 1
            # calculate Cx,j
            while j < size and load + tasks[j].loads <= info.capacity:
                load += tasks[j].loads
                # 这里算cost的方法还有问题，需要结合S1, S2, S3, S4
                cost += info.dist[tasks[j - 1].v, tasks[j].u] + tasks[j].cost
                curr_cost = cost + info.dist[task[j].v, info.depot]
                graph[i, j] = curr_cost
                j += 1
        return graph


    def shortest_path(self, graph, tasks):
        """ Dijkstra
            find the shortest path from task[0] to task[-1]
            找出起点到终点的最短路径，并且返回新的节点序
        """
        import heapq
        size = len(tasks)
        min_dist = np.copy(graph[0, :])
        pq = []
        for i in range(1, size):
            pq.append()
        for i in range(1, size):
            heapq.heappush(pq)


    def to_graph(self, tasks):
        info = self.info
        size = len(tasks)
        node_cnt = size * 2
        graph = np.full((node_cnt + 1, node_cnt + 1), -1)
        for i, task in enumerate(tasks):
            x = i * 2 + 1
            """ C1: 
                Given node (k + 1) on G* is a no-service node node 
                (i.e. it represents a no-service node on the giant tour), 
                then Ck,k+1 = 0.
            """
            # route(x-1, x) is no-service node
            graph[x, x - 1] = 0
            graph[x - 1, x] = 0

            """ C2 
                Ck,h is the sum of the node costs Ci,j,included in
                the vehicle tour and the fixed cost of the least
                capacity vehicle meeting the demand of the corresponding vehicle tour
            """
            # route(x, x+1) = {depot, x, x+1, depot}
            cost = task.cost + info.dist[info.depot, task.u] + info.dist[task.v, info.depot]
            graph[x, x + 1] = cost
            graph[x + 1, x] = cost

            load = task.load
            cost = info.dist[info.depot, task.u] + task.cost
            j = i + 1
            # calculate Cx,j
            while j < size and load + tasks[j].loads <= info.capacity:
                load += tasks[j].loads
                cost += info.dist[tasks[j - 1].v, tasks[j].u] + tasks[j].cost
                curr_cost = cost + info.dist[task[j].v, info.depot]
                graph[x, 2 * j + 2] = curr_cost
                graph[2 * j + 2, x] = curr_cost
                j += 1
        return graph


    def ulusoy_split(self, tasks):
        self.flatten(tasks)

    def get_path(self, graph):
        node_cost = np.zeros(len(graph) + 1)
        best_path = [[]] * (len(graph) + 1)
        node_cost[1] = np.inf
        for i in range(2, len(graph), 2):
            min_cost = np.inf
            best_node = []
            # find incoming node with min cost
            for node in incoming[i]:
                if node.cost < min_cost:
                    min_cost = node.cost
                    best_node = node
            if i % 2 == 0:
                pre_best_path = list(best_path[best_node.u])
                pre_best_path.append(best_node)
                best_path[i] = pre_best_path
            else:
                best_path[i] = best_path[i - 1]
            node_cost[i] += min_cost
            # release node, update outgoing node
            for node in outgoing[i]:
                idx = incoming[node.v].index(node)
                node.cost += node_cost[i]
                incoming[node.v, idx] = node
        return best_path[-1], node_cost[-1]

    def ulusoy_splitting(self, tasks):
        graph, incoming, outgoing = self.partition(tasks)
        return self.get_path(graph, incoming, outgoing)


class SBX:
    def __init__(self, info):
        self.info = info

    # finds the index where the route is best inserted
    def best_route_insertion(self, sub_route, route):
        start, _ = sub_route[0]
        _, end = sub_route[-1]
        best_payoff, best_i = 0, 0
        dist = self.info.dist
        for i in range(len(route) - 1):
            u1, v1 = route[i]
            u2, v2 = route[i + 1]
            init_cost = dist[v1, u2]
            payoff = init_cost - dist[u1, start] - dist[end, v2]
            if payoff > best_payoff:
                best_payoff, best_i = payoff, i
        return best_payoff, best_i

    # finds the best route index, and node index where the route should go
    def best_insert(self, child, sub_route):
        best_payoff, best_rid, best_nid = -1, 0, 0
        for r_id, route in enumerate(child):
            subopt_best, n_id = self.best_route_insertion(sub_route, route)
            if subopt_best > best_payoff:
                best_payoff, best_rid, best_nid = subopt_best, r_id, n_id
        return best_rid, best_nid

    def simple_random_crossover(self, ind1, ind2):
        child = copy.deepcopy(ind1)
        sub_route = ind2.random_subroute()
        for x in sub_route:
            child.remove(x)
        r_id, n_id = self.best_insert(child, sub_route)
        child.insert_route(r_id, sub_route)
        return child
