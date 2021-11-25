import copy
import random

import numpy as np
from definition import Solution, Insert, Move

MAX_TASK_TAG_LENGTH = 500
MAX_TASK_SEG_LENGTH = 550
MAX_TASK_SEQ_LENGTH = 500
task_routes = np.zeros((101, MAX_TASK_SEG_LENGTH), dtype=int)

MAX_NSIZE = 10  # upper bound of n size
MAX_ENSSIZE = 100  # max ENS neighborhood size


def find_element_position(position, arr, e):
    position[0] = 0
    pos = np.where(arr[1:arr[0] + 1] == e)[0]
    pos = pos + 1
    idx = np.arange(1, len(pos) + 1)
    position[idx] = pos
    position[0] = len(pos)
    return position
    # position[0] = 0
    # for i in range(1, arr[0] + 1):
    #     if arr[i] == e:
    #         position[0] += 1
    #         position[position[0]] = i




def del_element(arr, k):
    end = arr[0]
    arr[k:end] = arr[k + 1:end + 1]
    arr[end] = 0
    arr[0] -= 1
    return arr


def del_element_by_e(arr, e):
    idx = np.where(arr[1:arr[0] + 1] == e)
    idx = idx[0][0] + 1
    if idx < len(arr):
        arr[idx:arr[0]] = arr[idx + 1:arr[0] + 1]
    arr[-1] = 0
    arr[0] -= 1
    return arr


def add_element(arr, e, k):
    if k < 1 or k > arr[0] + 1:
        print('insert position error')
        return
    arr[k + 1:arr[0]+2] = arr[k:arr[0]+1]
    arr[k] = e
    return arr


def assign_subroute(route1, k1, k2, route2):
    # assign route1[k1:k2] to route2
    length = k2 - k1 + 1
    route2[0] = length
    route2[1:length + 1] = route1[k1:k2 + 1]
    return route2


def join_routes(route1, route2):
    # for i in range(1, route2[0] + 1):
    #     route1[0] += 1
    #     route1[route1[0]] = route2[i]
    # from_idx = route1[0] + 1
    length = route2[0]
    # to_idx = from_idx + length
    route1[route1[0] + 1:route1[0] + length + 1] = route2[1:length + 1]
    route1[0] = length
    return route1


def ind_route_converter(src: Solution, inst_task):
    load = 0
    task_seq = np.zeros(MAX_TASK_SEQ_LENGTH, dtype=int)
    loads = np.zeros(50, dtype=int)
    task_seq[0] = 1
    task_seq[1] = 0

    task_pointer = 1
    load_pointer = 0
    for task in src.task_seq[2:src.task_seq[0] + 1]:
        if task == 0:
            task_pointer += 1
            # dst.task_seq[task_pointer] = 0
            load_pointer += 1
            loads[load_pointer] = load
            load = 0
            continue
        load += inst_task[task].demand
        task_pointer += 1
        task_seq[task_pointer] = task
    task_seq[0] = task_pointer
    loads[0] = load_pointer
    return Solution(task_seq=task_seq, loads=loads, quality=src.quality, exceed_load=0)


class MAENS:
    def __init__(self, info):
        self.info = info
        # 种群大小
        self.psize = 30
        self.population = []
        # 初始化寻找种群的最大迭代数
        self.ubtrial = 50
        # 每次迭代生成的后代数量
        self.opsize = 6 * self.psize
        # Probability of carrying out local search (mutation)
        self.pls = 0.2
        # Number of routes involved in Merge-Split operator
        self.p = 2
        # Max number of generation
        self.Gmax = 500

        self.operations = [self.single_insertion, self.double_insertion, self.swap]

        from_head_to_depot = lambda x: self.info.min_dist[x, self.info.depot]
        self.rules = [
            lambda x, y, c: from_head_to_depot(x.v) > from_head_to_depot(y.v),
            lambda x, y, c: from_head_to_depot(x.v) < from_head_to_depot(y.v),
            lambda x, y, c: x.demand / x.cost > y.demand / y.cost,
            lambda x, y, c: x.demand / x.cost < y.demand / y.cost,
            lambda x, y, c: from_head_to_depot(x.v) > from_head_to_depot(y.v) if c < self.info.capacity / 2
            else from_head_to_depot(x.v) < from_head_to_depot(y.v)
        ]
        self.eq_rules = [
            lambda x, y, c: from_head_to_depot(x.v) == from_head_to_depot(y.v),
            lambda x, y, c: from_head_to_depot(x.v) == from_head_to_depot(y.v),
            lambda x, y, c: x.demand / x.cost == y.demand / y.cost,
            lambda x, y, c: x.demand / x.cost == y.demand / y.cost,
            lambda x, y, c: from_head_to_depot(x.v) == from_head_to_depot(y.v) if c < self.info.capacity / 2
            else from_head_to_depot(x.v) == from_head_to_depot(y.v)
        ]

    def rand_scanning(self, serve_mark):
        info = self.info
        inst_tasks = info.tasks

        serve_task_num = len(inst_tasks) - 1

        nearest_task = np.empty(serve_task_num + 1, dtype=int)
        sequence = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
        sequence[0] = 1
        sequence[1] = 0
        loads = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
        loads[0] = 0
        unserved_task = np.empty(serve_task_num + 1, dtype=int)
        candi_task = np.empty(serve_task_num + 1, dtype=int)

        for i, d in enumerate(np.where(serve_mark)[0]):
            unserved_task[i + 1] = d
        unserved_task[0] = serve_task_num

        load = 0
        trial = 0
        while trial < serve_task_num:
            if not unserved_task[0]:
                break
            curr_task = sequence[sequence[0]]
            counter = 0
            for i in range(1, unserved_task[0] + 1):
                if inst_tasks[unserved_task[i]].demand + load <= info.capacity:
                    counter += 1
                    candi_task[counter] = unserved_task[i]
            candi_task[0] = counter

            if candi_task[0] == 0:
                sequence[0] += 1
                sequence[sequence[0]] = 0
                loads[0] += 1
                loads[loads[0]] = load
                load = 0
                continue

            mindist = np.inf
            nearest_task[0] = 0

            for i in range(1, candi_task[0] + 1):
                curr_cost = info.min_dist[inst_tasks[curr_task].v, inst_tasks[candi_task[i]].u]
                if curr_cost < mindist:
                    mindist = curr_cost
                    nearest_task[0] = 1
                    nearest_task[nearest_task[0]] = candi_task[i]
                elif curr_cost == mindist:
                    nearest_task[0] += 1
                    nearest_task[nearest_task[0]] = candi_task[i]

            k = random.randrange(1, nearest_task[0] + 1)
            next_task = nearest_task[k]

            trial += 1
            sequence[0] += 1
            sequence[sequence[0]] = next_task
            load += inst_tasks[next_task].demand

            # find_element_position(positions, unserved_task, next_task)
            # del_element(unserved_task, positions[1])
            unserved_task = del_element_by_e(unserved_task, next_task)
            if inst_tasks[next_task].inverse > 0:
                unserved_task = del_element_by_e(unserved_task, inst_tasks[next_task].inverse)

        sequence[0] += 1
        sequence[sequence[0]] = 0

        loads[0] += 1
        loads[loads[0]] = load

        total_cost = self.get_task_seq_cost(sequence, inst_tasks)
        exceed_loads = self.get_exceed_loads(loads)
        return Solution(sequence, loads, total_cost, exceed_loads)

    def path_scanning(self, serve_mark):
        # min_cost, NRE, NRA, NVeh, capacity, is the extern variables.
        # info = self.info
        # task_num = info.edges_total
        # serve_task_num = np.sum(serve_mark[task_num+1:task_num])
        inst_tasks = self.info.tasks

        unserved_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        candidate_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        nearest_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        nearest_isol_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        nearest_inci_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        sel_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)

        i = 0
        for i, d in enumerate(np.where(serve_mark)[0]):
            unserved_task[i + 1] = d + 1
        unserved_task[0] = i

        # position = np.empty(MAX_TASK_SEG_LENGTH)
        # depot_dist = np.empty(MAX_TASK_TAG_LENGTH)
        # depot_dist = min_dist[info.depot, :].copy()
        # yeild = np.empty(MAX_TASK_TAG_LENGTH, dtype=np.float32)
        # yeild = np.array([inst_tasks[i].demand / inst_tasks[i].cost for i in np.where(serve_mark[:task_num+1])[0]])

        """Use five rules  to obtain solution"""
        # sequence = np.zeros(MAX_TASK_SEQ_LENGTH)
        # loads = np.zeros(101)
        # sequence[0] = 1
        #
        # cnt = 0
        # for i in np.where(serve_mark)[0]:
        #     cnt += 1
        #     unserved_task[cnt] = i
        # unserved_task[0] = cnt
        #
        # load = 0
        # trial = 0
        # while trial < serve_task_num:
        #     curr_task = sequence[sequence[0]]
        #
        #     cnt = 0
        #     for i in range(1, unserved_task[0] + 1):
        #         if inst_tasks[unserved_task[i]].demand <= info.capacity - load:
        #             cnt += 1
        #             candidate_task[cnt] = unserved_task[i]
        #     candidate_task[0] = cnt
        #
        #     if cnt == 0:
        #         sequence[0] += 1
        #         sequence[sequence[0]] = 0
        #         loads[0] += 1
        #         loads[loads[0]] = load
        #         load = 0
        #         continue
        #
        #     mindist = np.inf
        #     nearest_task[0] = 0
        #     for i in range(1, candidate_task[0] + 1):
        #         curr_cost = min_dist[inst_tasks[curr_task].v, inst_tasks[candidate_task[i]].u]
        #         if curr_cost < mindist:
        #             mindist = curr_cost
        #             nearest_task[0] = 1
        #             nearest_task[1] = candidate_task[i]
        #         elif curr_cost == mindist:
        #             nearest_task[0] += 1
        #             nearest_task[nearest_task[0]] = candidate_task[i]
        #
        #     nearest_inci_task[0] = 0
        #     nearest_isol_task[0] = 0
        #     for i in range(1, nearest_task[0]+1):
        #         if inst_tasks[nearest_task[i].v] == 1:
        #             nearest_inci_task[0] += 1
        #             nearest_inci_task[nearest_inci_task[0]] = nearest_task[i]
        #         else:
        #             nearest_isol_task[0] += 1
        #             nearest_isol_task[nearest_isol_task[0]] = nearest_task[i]
        #
        #     if nearest_isol_task[0] == 0:
        #         nearest_isol_task = nearest_inci_task[:nearest_inci_task[0]+1]
        #
        #     # for five phase, the above part is the same
        #     max_depot_dist = -1
        #     sel_task[0] = 0
        #     for task in nearest_isol_task[1:nearest_isol_task[0]+1]:
        #         if depot_dist[task] > max_depot_dist:
        #             max_depot_dist = depot_dist[task]
        #             sel_task[0] = 1
        #             sel_task[sel_task[0]] = task
        #         elif depot_dist[task] == max_depot_dist:
        #             sel_task[0] += 1
        #             sel_task[sel_task[0]] = task
        #     k = 1
        #     next_task = sel_task[k]
        #
        #     trial += 1
        #     sequence[0] += 1
        #     sequence[sequence[0]] = next_task
        #     load += inst_tasks[next_task].demand
        #
        #     # delete the served task in unserved task array
        #     del_element_by_e(unserved_task, next_task)
        #     if inst_tasks[next_task].inverse > 0:
        #         del_element_by_e(unserved_task, inst_tasks[next_task].inverse)
        #
        # sequence[0] += 1
        # sequence[sequence[0]] = 0
        # loads[0] += 1
        # loads[loads[0]] = load
        # quality = self.get_task_seq_cost(sequence, inst_tasks)
        # exceed_load = self.get_exceed_loads(loads)
        # tmp_ind = Solution(task_seq=sequence, loads=loads, quality=quality, exceed_load=exceed_load)
        tmp_sol_list = []
        for i in range(5):
            tmp_ind = self.path_scanning_with_rule(self.rules[i], self.eq_rules[i],
                                                   serve_mark, unserved_task, candidate_task,
                                                   nearest_task, nearest_inci_task, nearest_isol_task, sel_task)
            tmp_sol_list.append(tmp_ind)
        best_ind = min(tmp_sol_list, key=self.get_quality)
        return ind_route_converter(best_ind, inst_tasks)

    def get_quality(self, solution: Solution):
        return solution.quality

    def path_scanning_with_rule(self, rule, eq_rule,
                                serve_mark, unserved_task, candidate_task,
                                nearest_task, nearest_inci_task, nearest_isol_task, sel_task):
        info = self.info
        inst_tasks = info.tasks

        sequence = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
        loads = np.empty(101, dtype=int)
        sequence[0] = 1

        cnt = 0
        for i in np.where(serve_mark)[0]:
            cnt += 1
            unserved_task[cnt] = i
        unserved_task[0] = cnt

        load = 0
        trial = 0
        while trial < info.edges_total:
            curr_task = sequence[sequence[0]]

            cnt = 0
            for i in range(1, unserved_task[0] + 1):
                if inst_tasks[unserved_task[i]].demand <= info.capacity - load:
                    cnt += 1
                    candidate_task[cnt] = unserved_task[i]
            candidate_task[0] = cnt

            if cnt == 0:
                sequence[0] += 1
                sequence[sequence[0]] = 0
                loads[0] += 1
                loads[loads[0]] = load
                load = 0
                continue

            mindist = np.inf
            nearest_task[0] = 0
            for i in range(1, candidate_task[0] + 1):
                curr_cost = info.min_dist[inst_tasks[curr_task].v, inst_tasks[candidate_task[i]].u]
                if curr_cost < mindist:
                    mindist = curr_cost
                    nearest_task[0] = 1
                    nearest_task[1] = candidate_task[i]
                elif curr_cost == mindist:
                    nearest_task[0] += 1
                    nearest_task[nearest_task[0]] = candidate_task[i]

            nearest_inci_task[0] = 0
            nearest_isol_task[0] = 0
            for i in range(1, nearest_task[0] + 1):
                if inst_tasks[nearest_task[i].v] == 1:
                    nearest_inci_task[0] += 1
                    nearest_inci_task[nearest_inci_task[0]] = nearest_task[i]
                else:
                    nearest_isol_task[0] += 1
                    nearest_isol_task[nearest_isol_task[0]] = nearest_task[i]

            if nearest_isol_task[0] == 0:
                nearest_isol_task = nearest_inci_task[:nearest_inci_task[0] + 1]
            # for five phase, the above part is the same

            best = nearest_isol_task[1]
            sel_task[0] = 1
            sel_task[1] = best
            for task in nearest_isol_task[2:nearest_isol_task[0] + 1]:
                if self.is_better(inst_tasks(task), inst_tasks(best), load, rule):
                    best = task
                    sel_task[0] = 1
                    sel_task[1] = task
                elif self.is_equal(inst_tasks(task), inst_tasks(best), load, eq_rule):
                    sel_task[0] += 1
                    sel_task[sel_task[0]] = task
            k = 1
            next_task = sel_task[k]

            trial += 1
            sequence[0] += 1
            sequence[sequence[0]] = next_task
            load += inst_tasks[next_task].demand

            # delete the served task in unserved task array
            unserved_task = del_element_by_e(unserved_task, next_task)
            if inst_tasks[next_task].inverse > 0:
                unserved_task = del_element_by_e(unserved_task, inst_tasks[next_task].inverse)

        sequence[0] += 1
        sequence[sequence[0]] = 0
        loads[0] += 1
        loads[loads[0]] = load
        quality = self.get_task_seq_cost(sequence, inst_tasks)
        # exceed_load = self.get_exceed_loads(loads)
        exceed_load = 0
        return Solution(task_seq=sequence, loads=loads, quality=quality, exceed_load=exceed_load)

    def is_better(self, curr, prev, current_load, rule):
        return rule(curr, prev, current_load)

    def is_equal(self, curr, prev, current_load, rule):
        return rule(curr, prev, current_load)

    def get_task_seq_cost(self, task_seq, inst_tasks):
        total_cost = 0
        min_dist = self.info.min_dist
        for i in range(1, task_seq[0]):
            total_cost += min_dist[inst_tasks[task_seq[i]].v, inst_tasks[task_seq[i + 1]].u] \
                          + inst_tasks[task_seq[i]].cost
        return total_cost

    def get_exceed_loads(self, route_seg_load):
        exceed_load = 0
        capacity = self.info.capacity
        for load in route_seg_load[1:route_seg_load[0] + 1]:
            exceed_load += max(0, load - capacity)
        return exceed_load

    def SBX(self, s1, s2):
        """ Sequence Based Crossover Operator
            S1 -> {R11, R12}
            S2 -> {R21, R22}
            Sx -> {任意拼接}
            remove duplicate task in NewR and insert missing task to NewR -> offspring
        """
        info = self.info
        inst_tasks = info.tasks
        min_dist = info.min_dist

        # max_task_seq_length = info.edges_total
        sub_path1 = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
        sub_path2 = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
        routes1 = np.empty((101, MAX_TASK_SEQ_LENGTH), dtype=int)
        routes2 = np.empty((101, MAX_TASK_SEQ_LENGTH), dtype=int)

        """ define: candidate selection:
                (routeID, position)
        """
        candidate_list1, candidate_list2 = [], []

        position = np.zeros(101, dtype=int)
        xclds = np.empty(101, dtype=int)
        left_tasks = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)

        position = find_element_position(position, s1.task_seq, 0)
        routes1[0, 0] = position[0] - 1
        for i in range(1, position[0]):
            routes1[i] = assign_subroute(s1.task_seq, position[i], position[i + 1], routes1[i])

        position = find_element_position(position, s2.task_seq, 0)
        routes2[0, 0] = position[0] - 1
        for i in range(1, position[0]):
            routes2[i] = assign_subroute(s2.task_seq, position[i], position[i + 1], routes2[i])

        xclds[:s1.loads[0] + 1] = s1.loads[:s1.loads[0] + 1]

        for i in range(1, routes1[0, 0] + 1):
            for j in range(2, routes1[i, 0]):
                candidate_list1.append((i, j))

        for i in range(1, routes2[0, 0] + 1):
            for j in range(2, routes2[i, 0]):
                candidate_list2.append((i, j))

        k1 = random.randrange(1, len(candidate_list1))
        k2 = random.randrange(1, len(candidate_list2))

        sel_routeID_1, sel_pos_1 = candidate_list1[k1]
        sel_routeID_2, sel_pos_2 = candidate_list2[k2]
        sub_path1 = assign_subroute(routes1[sel_routeID_1], 1, sel_pos_1, sub_path1)
        sub_path2 = assign_subroute(routes2[sel_routeID_2], sel_pos_2,
                                    routes2[sel_routeID_2, 0], sub_path2)
        left_tasks = assign_subroute(routes1[sel_routeID_1], sel_pos_1 + 1,
                                     routes1[sel_routeID_1, 0] - 1, left_tasks)

        # remove duplicated task for routes1
        checked = np.zeros(MAX_TASK_SEQ_LENGTH, dtype=int)
        for i in range(1, sub_path2[0]):
            # if checked[i]:
            #     continue
            for j in range(sub_path1[0], 1, -1):
                if sub_path1[j] == sub_path2[i] or sub_path1[j] == inst_tasks[sub_path2[i]].inverse:
                    sub_path1 = del_element(sub_path1, j)
                    checked[i] = 1
                    break

        for i in range(1, sub_path2[0]):
            if checked[i]:
                continue
            for j in range(left_tasks[0], 0, -1):
                if left_tasks[j] == sub_path2[i] or left_tasks[j] == inst_tasks[sub_path2[i]].inverse:
                    left_tasks = del_element(left_tasks, j)
                    checked[i] = 1
                    break

        for i in range(1, sub_path2[0]):
            if checked[i]:
                continue
            for j in range(1, routes1[0, 0] + 1):
                if j == candidate_list1[k1][0]:
                    continue
                for k in range(routes1[j, 0], 1, -1):
                    if routes1[j, k] == sub_path2[i] or routes1[j, k] == inst_tasks[sub_path2[i]].inverse:
                        routes1[j] = del_element(routes1[j], k)
                        xclds[j] -= inst_tasks[sub_path2[i]].demand
                        checked[i] = 1

                if checked[i]:
                    break

        sub_path1 = join_routes(sub_path1, sub_path2)

        sel_routeID = candidate_list1[k1][0]
        routes1[sel_routeID] = sub_path1
        xclds[sel_routeID] = 0
        for i in range(2, routes1[sel_routeID, 0]):
            xclds[sel_routeID] += inst_tasks[routes1[sel_routeID, i]].demand

        left_tasks_cnt = left_tasks[0]

        # insert missing tasks
        candidate_insertions = [[]] * 6000
        pare_to_set_insertions = [[]] * 6000

        curr_route = 0
        for j in range(1, routes1[0, 0] + 1):
            if routes1[j, 0] == 2:
                continue
            curr_route += 1

        out = np.empty(6000, dtype=int)
        for n in range(1, left_tasks_cnt + 1):
            candidate_insert_cnt = 0
            pare_to_set_size = 0
            curr_task = 0
            for j in range(1, routes1[0, 0]):
                if routes1[j, 0] == 2:
                    continue
                curr_task = left_tasks[n]
                if xclds[j] > info.capacity:
                    ivload = inst_tasks[curr_task].demand
                elif xclds[j] > info.capacity - inst_tasks[curr_task].demand:
                    ivload = xclds[j] + inst_tasks[curr_task].demand - info.capacity
                else:
                    ivload = 0

                for k in range(2, routes1[j, 0] + 1):
                    candidate_insert_cnt += 1
                    insert_cost = min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[curr_task].u] \
                                  + min_dist[inst_tasks[curr_task].v, inst_tasks[routes1[j, k]].u] \
                                  - min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[routes1[j, k]].u]
                    candidate_insertions[candidate_insert_cnt] = Insert(task=curr_task, routeID=j, position=k,
                                                                        cost=insert_cost, exceed_load=ivload)

                    out[0] = 0
                    add = 1
                    for m in range(1, pare_to_set_size + 1):
                        if candidate_insertions[candidate_insert_cnt].cost > pare_to_set_insertions[m].cost and \
                                candidate_insertions[candidate_insert_cnt].exceed_load > pare_to_set_insertions[m].exceed_load:
                            add = 0
                            break
                        elif candidate_insertions[candidate_insert_cnt].cost < pare_to_set_insertions[m].cost and \
                                candidate_insertions[candidate_insert_cnt].exceed_load < pare_to_set_insertions[
                            m].exceed_load:
                            out[0] += 1
                            out[out[0]] = m
                    if add:
                        for m in range(out[0], 0, -1):
                            pare_to_set_insertions[out[m]:pare_to_set_size] = pare_to_set_insertions[
                                                                              out[m] + 1:pare_to_set_size + 1]
                            pare_to_set_size -= 1
                        pare_to_set_size += 1
                        pare_to_set_insertions[pare_to_set_size] = candidate_insertions[candidate_insert_cnt]
                    w = inst_tasks[curr_task].inverse
                    candidate_insert_cnt += 1

                    insert_cost = min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[curr_task].u] \
                                  + min_dist[inst_tasks[w].v, inst_tasks[routes1[j, k]].u] \
                                  - min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[routes1[j, k]].u]
                    candidate_insertions[candidate_insert_cnt] = Insert(task=w, routeID=j, position=k, cost=insert_cost,
                                                                        exceed_load=ivload)

                    out[0] = 0
                    add = 1
                    for m in range(1, pare_to_set_size + 1):
                        if candidate_insertions[candidate_insert_cnt].cost > pare_to_set_insertions[m].cost and \
                                candidate_insertions[candidate_insert_cnt].exceed_load > pare_to_set_insertions[
                            m].exceed_load:
                            add = 0
                            break
                        elif candidate_insertions[candidate_insert_cnt].cost < pare_to_set_insertions[m].cost and \
                                candidate_insertions[candidate_insert_cnt].exceed_load < pare_to_set_insertions[
                            m].exceed_load:
                            out[0] += 1
                            out[out[0]] = m
                    if add:
                        for m in range(out[0], 0, -1):
                            pare_to_set_insertions[out[m]:pare_to_set_size] = pare_to_set_insertions[
                                                                              out[m] + 1:pare_to_set_size + 1]
                            pare_to_set_size -= 1
                        pare_to_set_size += 1
                        pare_to_set_insertions[pare_to_set_size] = candidate_insertions[candidate_insert_cnt]

            # insert as a new route
            candidate_insert_cnt += 1
            insert_cost = min_dist[info.depot, inst_tasks[curr_task].u] \
                          + min_dist[inst_tasks[curr_task].v, info.depot]
            candidate_insertions[candidate_insert_cnt] = Insert(task=curr_task, routeID=0, position=2, cost=insert_cost,
                                                                exceed_load=0)

            out[0] = 0
            add = 1
            for m in range(1, pare_to_set_size + 1):
                if candidate_insertions[candidate_insert_cnt].cost > pare_to_set_insertions[m].cost and \
                        candidate_insertions[candidate_insert_cnt].exceed_load > pare_to_set_insertions[m].exceed_load:
                    add = 0
                    break
                elif candidate_insertions[candidate_insert_cnt].cost < pare_to_set_insertions[m].cost and \
                        candidate_insertions[candidate_insert_cnt].exceed_load < pare_to_set_insertions[m].exceed_load:
                    out[0] += 1
                    out[out[0]] = m
            if add:
                for m in range(out[0], 0, -1):
                    pare_to_set_insertions[out[m]:pare_to_set_size] = pare_to_set_insertions[
                                                                      out[m] + 1:pare_to_set_size + 1]
                    pare_to_set_size -= 1
                pare_to_set_size += 1
                pare_to_set_insertions[pare_to_set_size] = candidate_insertions[candidate_insert_cnt]

            k = random.randrange(1, pare_to_set_size + 1)
            best_insertion = pare_to_set_insertions[k]
            if best_insertion.routeID == 0:
                routes1[0, 0] += 1
                tot_cnt = routes1[0, 0]
                routes1[tot_cnt, 0] = 3
                routes1[tot_cnt, 1] = 0
                routes1[tot_cnt, 2] = best_insertion.task
                routes1[tot_cnt, 3] = 0

                xclds[0] += 1
                xclds[xclds[0]] = inst_tasks[best_insertion.task].demand
            else:
                routes1[best_insertion.routeID] = add_element(routes1[best_insertion.routeID], best_insertion.task, best_insertion.position)
                xclds[best_insertion.routeID] += inst_tasks[best_insertion.task].demand

        # transfer routes1 to sequence
        sequence = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
        sequence[0] = 1
        for i in range(1, routes1[0, 0] + 1):
            if routes1[i, 0] == 2:
                continue
            sequence[0] -= 1
            sequence = join_routes(sequence, routes1[i])

        loads = xclds
        for i in range(loads[0], 0, -1):
            if loads[i] == 0:
                loads = del_element(loads, i)

        quality = self.get_task_seq_cost(sequence, inst_tasks)
        exceed_load = self.get_exceed_loads(loads)
        return Solution(sequence, loads, quality, exceed_load)

    def lns_mut(self, sx: Solution, best_fsb_solution: Solution):
        info = self.info

        sls = copy.deepcopy(sx)
        coef = best_fsb_solution.quality / info.capacity * (
                best_fsb_solution.quality / sx.quality + sx.exceed_load / info.capacity + 1.0)
        sls.calculate_fitness(coef)

        count, count1 = 0, 0
        count_fsb, count_infsb = 0, 0

        imp = 1  # improved
        count1 = 0
        while imp:
            count += 1
            count1 += 1
            imp = 0

            if sls.exceed_load == 0:
                count_fsb += 1
            else:
                count_infsb += 1

            if count % 5 == 0:
                # 系数随着迭代次数和可行解数量改变
                if count_fsb == 5:
                    coef /= 5
                    sls.calculate_fitness(coef)
                elif count_infsb == 5:
                    coef *= 5
                    sls.calculate_fitness(coef)
                count_fsb = 0
                count_infsb = 0

            # tmp_ind = copy.deepcopy(sls)
            prev_fitness = sls.fitness
            prev_quality = sls.quality
            # traditional operator
            sls = self.lns(sls, coef, 1)
            if sls.fitness < prev_fitness:
                imp = 1
            if count1 > 50 and sls.quality < prev_quality:
                break
            if sls.exceed_load == 0 and sls.quality < best_fsb_solution.quality:
                best_fsb_solution = copy.deepcopy(sls)

        imp = 1  # improved
        count1 = 0
        while imp:
            count += 1
            count1 += 1
            imp = 0

            if sls.exceed_load == 0:
                count_fsb += 1
            else:
                count_infsb += 1

            if count % 5 == 0:
                # 系数随着迭代次数和可行解数量改变
                if count_fsb == 5:
                    coef /= 5
                    sls.calculate_fitness(coef)
                elif count_infsb == 5:
                    coef *= 5
                    sls.calculate_fitness(coef)
                count_fsb = 0
                count_infsb = 0

            # tmp_ind = copy.deepcopy(sls)
            prev_fitness = sls.fitness
            prev_quality = sls.quality
            # traditional operator
            sls = self.lns(sls, coef, 2)
            if sls.fitness < prev_fitness:
                imp = 1
            if count1 > 50 and sls.quality < prev_quality:
                break
            if sls.exceed_load == 0 and sls.quality < best_fsb_solution.quality:
                best_fsb_solution = copy.deepcopy(sls)

        imp = 1  # improved
        count1 = 0
        while imp:
            count += 1
            count1 += 1
            imp = 0

            if sls.exceed_load == 0:
                count_fsb += 1
            else:
                count_infsb += 1

            if count % 5 == 0:
                # 系数随着迭代次数和可行解数量改变
                if count_fsb == 5:
                    coef /= 5
                    sls.calculate_fitness(coef)
                elif count_infsb == 5:
                    coef *= 5
                    sls.calculate_fitness(coef)
                count_fsb = 0
                count_infsb = 0

            tmp_ind = copy.deepcopy(sls)
            # traditional operator
            sls = self.lns(sls, coef, 1)
            if sls.fitness < tmp_ind.fitness:
                imp = 1
            if count1 > 50 and sls.quality < tmp_ind.quality:
                break
            if sls.exceed_load == 0 and sls.quality < best_fsb_solution.quality:
                best_fsb_solution = copy.deepcopy(sls)
        return sls, best_fsb_solution

    def get_fitness(self, move: Move):
        return move.fitness

    def lns(self, ind: Solution, coef, nsize):
        info = self.info
        inst_tasks = info.tasks
        ind.calculate_fitness(coef)

        if nsize == 1:
            # traditional move
            next_move: Move = min([move(ind, coef) for move in self.operations], key=self.get_fitness)

            # 将task_route转化为ind.task_seq
            # end_points = np.where(ind.task_seq[1:ind.task_seq[0] + 1] == 0)[0] + 1
            # orig_ptr = end_points[next_move.orig_seg - 1] + next_move.orig_pos
            # if next_move.targ_seg < len(end_points):
            #     targ_ptr = end_points[next_move.targ_seg - 1] + next_move.targ_seg
            # else:
            #     # ind.task_seq[ind.task_seq[0] + 1] = 0
            #     targ_ptr = ind.task_seq[0] + 1
            orig_ptr, targ_ptr = 0, 0
            seg_ptr1, seg_ptr2 = 0, 0
            for i in range(1, ind.task_seq[0]):
                if ind.task_seq[i] == 0:
                    if seg_ptr1 < next_move.orig_seg:
                        seg_ptr1 += 1
                    if seg_ptr2 < next_move.targ_seg:
                        seg_ptr2 += 1
                    if seg_ptr1 == next_move.orig_seg and orig_ptr == 0:
                        orig_ptr = i + next_move.orig_pos - 1
                    if seg_ptr2 == next_move.targ_seg and targ_ptr == 0:
                        targ_ptr = i + next_move.targ_pos - 1
                if orig_ptr != 0 and targ_ptr != 0:
                    break

            if next_move.type == 1:
                # single insertion
                ind.task_seq = del_element(ind.task_seq, orig_ptr)
                if targ_ptr > orig_ptr:
                    targ_ptr -= 1
                ind.loads[next_move.orig_seg] -= inst_tasks[next_move.task1].demand

                if next_move.targ_seg > ind.loads[0]:
                    ind.task_seq[0] += 2
                    ind.task_seq[ind.task_seq[0] - 1] = next_move.task1
                    ind.task_seq[ind.task_seq[0]] = 0
                    ind.loads[0] += 1
                    ind.loads[ind.loads[0]] = inst_tasks[next_move.task1].demand
                else:
                    ind.task_seq = add_element(ind.task_seq, next_move.task1, targ_ptr)
                    ind.loads[next_move.targ_seg] += inst_tasks[next_move.task1].demand
            elif next_move.type == 2:
                # doubel insertion
                ind.task_seq = del_element(ind.task_seq, orig_ptr + 1)
                ind.task_seq = del_element(ind.task_seq, orig_ptr)
                if targ_ptr > orig_ptr:
                    targ_ptr -= 2

                ind.loads[next_move.orig_seg] -= inst_tasks[next_move.task1].demand + inst_tasks[next_move.task2].demand
                if next_move.targ_seg > ind.loads[0]:
                    ind.task_seq[0] += 3
                    ind.task_seq[ind.task_seq[0] - 2] = next_move.task1
                    ind.task_seq[ind.task_seq[0] - 1] = next_move.task2
                    ind.task_seq[ind.task_seq[0]] = 0
                    ind.loads[0] += 1
                    ind.loads[ind.loads[0]] = inst_tasks[next_move.task1].demand + inst_tasks[next_move.task2].demand
                else:
                    ind.task_seq = add_element(ind.task_seq, next_move.task2, targ_ptr)
                    ind.task_seq = add_element(ind.task_seq, next_move.task1, targ_ptr)
                    ind.loads[next_move.targ_seg] += inst_tasks[next_move.task1].demand + inst_tasks[
                        next_move.task2].demand
            elif next_move.type == 3:
                ind.task_seq[targ_ptr] = next_move.task1
                ind.task_seq[orig_ptr] = next_move.task2
                ind.loads[next_move.orig_seg] -= inst_tasks[next_move.task1].demand - inst_tasks[next_move.task2].demand
                ind.loads[next_move.targ_seg] += inst_tasks[next_move.task1].demand - inst_tasks[next_move.task2].demand

            ind.quality = next_move.quality
            ind.exceed_load = next_move.exceed_load
            ind.fitness = next_move.fitness
            if ind.loads[next_move.orig_seg] == 0:
                if next_move.type == 2 and next_move.orig_seg > next_move.targ_seg:
                    ind.task_seq = del_element(ind.task_seq, orig_ptr + 1)
                else:
                    ind.task_seq = del_element(ind.task_seq, orig_ptr)
                ind.loads = del_element(ind.loads, next_move.orig_seg)
        else:
            # M-S operator (merge and split)
            task_routes[0, 0] = 1
            task_routes[1, 0] = 1
            task_routes[1, 1] = 0
            for i in range(2, ind.task_seq[0] + 1):
                task_routes[task_routes[0, 0], 0] += 1
                task_routes[task_routes[0, 0], task_routes[task_routes[0, 0], 0]] = ind.task_seq[i]

                if ind.task_seq[i] == 0 and i < ind.task_seq[0]:
                    task_routes[0, 0] += 1
                    task_routes[task_routes[0, 0], 0] = 1
                    task_routes[task_routes[0, 0], 1] = 0

            if task_routes[0, 0] < nsize:
                return

            multi = task_routes[0, 0]
            ub_trial = multi
            for i in range(1, nsize):
                multi -= 1
                ub_trial *= multi
            multi = nsize
            for i in range(1, nsize):
                ub_trial //= multi
                multi -= 1

            maxcount = min(ub_trial, MAX_ENSSIZE)

            candidate_combs = np.zeros((MAX_NSIZE + 1, MAX_ENSSIZE + 1), dtype=int)
            pointers = np.arange(0, nsize + 1, 1)

            for i in range(1, maxcount + 1):
                candidate_combs[i, 0] = nsize
                for j in range(1, nsize + 1):
                    candidate_combs[i, j] = pointers[j]

                curr_ptr = nsize
                while pointers[curr_ptr] == task_routes[0, 0] - nsize + curr_ptr:
                    curr_ptr -= 1
                if curr_ptr == 0:
                    break

                pointers[curr_ptr] += 1
                for j in range(curr_ptr + 1, nsize + 1):
                    pointers[j] = pointers[j - 1] + 1

            for i in range(maxcount):
                lns_routes = candidate_combs[i].copy()

                sel_total_load = 0
                for j in range(1, lns_routes[0] + 1):
                    sel_total_load += ind.loads[lns_routes[j]]

                if sel_total_load > nsize * info.capacity:
                    continue

                serve_mark = np.zeros(MAX_TASK_TAG_LENGTH, dtype=int)
                serve_mark[0] = info.edges_required
                for j in range(1, lns_routes[0] + 1):
                    for k in range(2, task_routes[lns_routes[j, 0]]):
                        serve_mark[task_routes[lns_routes[j, k]]] = 1
                        serve_mark[inst_tasks[task_routes[lns_routes[j], k]].inverse] = 1

                self.path_scanning(serve_mark)
        return ind

    def single_insertion(self, ind: Solution, coef):
        """
        Move operator: Single Insertion
         A task is removed from its current position
         and re-inserted into another position of the current solution or a new empty route.

         If the selected task belongs to an edge task, both its directions will
         be considered when inserting the task into the “target position.”

        :param ind: parent
        :return: move generated by Single Insertion
        """
        info = self.info
        inst_tasks = info.tasks
        min_dist = info.min_dist

        """将individual 的task_seq转化为task_route格式
        task_routes[0, 0]: #routes
        task_routes[i, 0]: routes[i]的 #tasks
        task_toutes[i, j]: routes[i], task[j]在inst_task中对应的编号
        
        task_routes[i, 0] == 1: 
        
        ind.task_seq[i] == 0: 一条route的终点（下一条route起点）
                        != 0: route上的task在inst_task中的编号
        """
        task_routes[0, 0] = 1
        task_routes[1, 0] = 1
        task_routes[1, 1] = 0
        for i in range(2, ind.task_seq[0] + 1):
            task_routes[task_routes[0, 0], 0] += 1
            task_routes[task_routes[0, 0], task_routes[task_routes[0, 0], 0]] = ind.task_seq[i]
            if ind.task_seq[i] == 0 and i < ind.task_seq[0]:
                task_routes[0, 0] += 1
                task_routes[task_routes[0, 0], 0] = 1
                task_routes[task_routes[0, 0], 1] = 0

        best_move = Move(type=1)
        tmp_move = Move(type=1)
        # task_routes[0, 0]: segment总数
        for s1 in range(1, task_routes[0, 0] + 1):
            # orig_seg: 被remove的task所在的segment
            orig_seg = s1
            # task_routes[seg, 0]: segment中的路径条数
            for i in range(2, task_routes[s1, 0]):
                orig_pos = i
                for s2 in range(1, task_routes[0, 0] + 2):
                    if s2 == s1:
                        continue
                    # s2: 插入的segment
                    targ_seg = s2
                    if s2 > task_routes[0, 0]:
                        exceed_load = ind.exceed_load
                        if ind.loads[s1] > info.capacity:
                            exceed_load -= ind.loads[s1] - info.capacity
                        if ind.loads[s1] - inst_tasks[task_routes[s1, i]].demand > info.capacity:
                            exceed_load += ind.loads[i] - inst_tasks[task_routes[s1, i]].demand - info.capacity

                        task1 = task_routes[s1, i]
                        quality = ind.quality + min_dist[
                            inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  + min_dist[info.depot, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, info.depot] \
                                  - min_dist[
                                      inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[
                                      inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load
                        continue

                    for j in range(2, task_routes[s2, 0] + 1):
                        if inst_tasks[task_routes[s2, j - 1]].v == inst_tasks[task_routes[s2, j]].u:
                            continue
                        # targ_pos：插入的位置
                        targ_pos = j
                        exceed_load = ind.exceed_load
                        # 计算新route的load
                        exceed_load -= max(ind.loads[s1] - info.capacity, 0)
                        exceed_load -= max(ind.loads[s2] - info.capacity, 0)
                        exceed_load += max(ind.loads[s1] - inst_tasks[task_routes[s1, i]].demand - info.capacity, 0)
                        exceed_load += max(ind.loads[s2] + inst_tasks[task_routes[s1, i]].demand - info.capacity, 0)

                        task1 = task_routes[s1, i]
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = inst_tasks[task_routes[s1, i]].inverse
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load
        return best_move

    def double_insertion(self, ind: Solution, coef):
        """double insertion:
        similar to single insertion.
        two consecutive tasks are moved instead of a single task
        :return: best move
        """
        info = self.info
        inst_tasks = info.tasks
        min_dist = info.min_dist

        task_routes[0, 0] = 1
        task_routes[1, 0] = 1
        task_routes[1, 1] = 0
        for i in range(2, ind.task_seq[0] + 1):
            task_routes[task_routes[0, 0], 0] += 1
            task_routes[task_routes[0, 0], task_routes[task_routes[0, 0], 0]] = ind.task_seq[i]
            if ind.task_seq[i] == 0 and i < ind.task_seq[0]:
                task_routes[0, 0] += 1
                task_routes[task_routes[0, 0], 0] = 1
                task_routes[task_routes[0, 0], 1] = 0

        best_move = Move(type=2)
        tmp_move = Move(type=2)
        for s1 in range(1, task_routes[0, 0] + 1):
            if task_routes[s1, 0] < 4:
                continue

            orig_seg = s1
            for i in range(2, task_routes[s1, 0] - 1):
                orig_pos = i
                for s2 in range(1, task_routes[0, 0] + 2):
                    if s2 == s1:
                        continue
                    targ_seg = s2
                    if s2 > task_routes[0, 0]:
                        if task_routes[s1, 0] <= 4:
                            continue
                        exceed_load = ind.exceed_load
                        exceed_load -= max(ind.loads[s1] - info.capacity, 0)
                        exceed_load += max(ind.loads[i] - inst_tasks[task_routes[s1, i]].demand \
                                           - inst_tasks[task_routes[s1, i + 1]].demand \
                                           - info.capacity, 0)
                        exceed_load += max(inst_tasks[task_routes[s1, i]].demand \
                                           + inst_tasks[task_routes[s1, i + 1]].demand \
                                           - info.capacity, 0)

                        task1 = task_routes[s1, i]
                        task2 = task_routes[s1, i + 1]
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  + min_dist[info.depot, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, info.depot]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = task_routes[s1, i]
                        task2 = inst_tasks[task_routes[s1, i + 1]].inverse

                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  + min_dist[info.depot, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, info.depot]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load
                        continue

                    for j in range(2, task_routes[s2, 0] + 1):
                        if inst_tasks[task_routes[s2, j - 1]].v == inst_tasks[task_routes[s2, j]].u:
                            continue
                        targ_pos = j
                        exceed_load = ind.exceed_load
                        exceed_load -= max(ind.loads[s1] - info.capacity, 0)
                        exceed_load -= max(ind.loads[s2] - info.capacity, 0)
                        exceed_load += max(ind.loads[s1] \
                                           - inst_tasks[task_routes[s1, i]].demand \
                                           - inst_tasks[task_routes[s1, i + 1]].demand \
                                           - info.capacity, 0)
                        exceed_load += max(ind.loads[s2] \
                                           + inst_tasks[task_routes[s1, i]].demand \
                                           + inst_tasks[task_routes[s1, i + 1]].demand \
                                           - info.capacity, 0)

                        task1 = task_routes[s1, i]
                        task2 = task_routes[s1, i + 1]

                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = inst_tasks[task_routes[s1, i]].inverse
                        task2 = task_routes[s1, i + 1]
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = task_routes[s1, i]
                        task2 = inst_tasks[task_routes[s1, i + 1]].inverse
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = inst_tasks[task_routes[s1, i]].inverse
                        task2 = inst_tasks[task_routes[s1, i + 1]].inverse
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load
        return best_move

    def swap(self, ind: Solution, coef):
        info = self.info
        inst_tasks = info.tasks
        min_dist = info.min_dist

        task_routes[0, 0] = 1
        task_routes[1, 0] = 1
        task_routes[1, 1] = 0
        for i in range(2, ind.task_seq[0] + 1):
            task_routes[task_routes[0, 0], 0] += 1
            task_routes[task_routes[0, 0], task_routes[task_routes[0, 0], 0]] = ind.task_seq[i]
            if ind.task_seq[i] == 0 and i < ind.task_seq[0]:
                task_routes[0, 0] += 1
                task_routes[task_routes[0, 0], 0] = 1
                task_routes[task_routes[0, 0], 1] = 0

        best_move = Move(type=3)
        tmp_move = Move(type=3)
        for s1 in range(1, task_routes[0, 0] + 1):
            orig_seg = s1
            for i in range(2, task_routes[s1, 0] - 1):
                if inst_tasks[task_routes[s1, i - 1]].v == inst_tasks[task_routes[s1, i]].u and \
                        inst_tasks[task_routes[s1, i]].v == inst_tasks[task_routes[s1, i + 1]].u:
                    continue
                orig_pos = i
                for s2 in range(s1 + 1, task_routes[0, 0] + 1):
                    targ_seg = s2
                    for j in range(2, task_routes[s2, 0]):
                        if inst_tasks[task_routes[s2, j - 1]].v == inst_tasks[task_routes[s2, j]].u and \
                                inst_tasks[task_routes[s2, j]].v == inst_tasks[task_routes[s2, j + 1]].u:
                            continue
                        targ_pos = j
                        exceed_load = ind.exceed_load
                        exceed_load -= max(ind.loads[s1] - info.capacity, 0)
                        exceed_load -= max(ind.loads[s2] - info.capacity, 0)
                        exceed_load += max(ind.loads[s1] \
                                           - inst_tasks[task_routes[s1, i]].demand \
                                           + inst_tasks[task_routes[s2, j]].demand \
                                           - info.capacity, 0)
                        exceed_load += max(ind.loads[s2] \
                                           + inst_tasks[task_routes[s1, i]].demand \
                                           - inst_tasks[task_routes[s2, j]].demand \
                                           - info.capacity, 0)

                        task1 = task_routes[s1, i]
                        task2 = task_routes[s2, j]

                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task_routes[s2, j + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j]].v, inst_tasks[task_routes[s2, j + 1]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = inst_tasks[task_routes[s1, i]].inverse
                        task2 = task_routes[s2, j]
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task_routes[s2, j + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j]].v, inst_tasks[task_routes[s2, j + 1]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = task_routes[s1, i]
                        task2 = inst_tasks[task_routes[s1, i + 1]].inverse
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task_routes[s2, j + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j]].v, inst_tasks[task_routes[s2, j + 1]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load

                        task1 = inst_tasks[task_routes[s1, i]].inverse
                        task2 = inst_tasks[task_routes[s1, i + 1]].inverse
                        quality = ind.quality \
                                  + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task2].u] \
                                  + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                  - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                  + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                  + min_dist[inst_tasks[task1].v, inst_tasks[task_routes[s2, j + 1]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u] \
                                  - min_dist[inst_tasks[task_routes[s2, j]].v, inst_tasks[task_routes[s2, j + 1]].u]
                        tmp_move.quality = quality
                        tmp_move.exceed_load = exceed_load
                        tmp_move.calculate_fitness(coef)

                        if tmp_move.fitness < best_move.fitness:
                            best_move.fitness = tmp_move.fitness
                            best_move.task1 = task1
                            best_move.task2 = task2
                            best_move.orig_seg = orig_seg
                            best_move.targ_seg = targ_seg
                            best_move.orig_pos = orig_pos
                            best_move.targ_pos = targ_pos
                            best_move.quality = quality
                            best_move.exceed_load = exceed_load
        return best_move

    def initialize(self):
        task_num = len(self.info.tasks) - 1
        trial = 0
        population = []
        best_fsb_solution = Solution(None, None, np.inf, 0)
        # population.append(best_fsb_solution)
        while trial < self.ubtrial and len(population) < self.psize:
            trial += 1
            serve_mark = np.zeros(MAX_TASK_TAG_LENGTH, dtype=int)
            serve_mark[1:task_num + 1] = 1
            # serve_mark = np.append(np.ones(task_num, dtype=int), np.zeros(MAX_TASK_TAG_LENGTH - task_num, dtype=int))
            init_ind = self.rand_scanning(serve_mark)
            if init_ind in population and trial != self.ubtrial:
                continue
            population.append(init_ind)
            if init_ind.exceed_load == 0 and init_ind.quality < best_fsb_solution.quality:
                best_fsb_solution = init_ind
        self.psize = len(population)
        self.opsize = 6 * self.psize
        self.population = [[]] * (self.psize + self.opsize)
        self.population[:self.psize] = population
        return best_fsb_solution

    def random_select(self):
        s1_idx = random.randrange(0, self.psize)
        s2_idx = random.randrange(0, self.psize)
        while s1_idx == s2_idx:
            s2_idx = random.randrange(1, self.psize)
        s1 = self.population[s1_idx]
        s2 = self.population[s2_idx]
        return s1, s2

    def stochastic_rank(self):
        pf = 0.45
        total_size = self.psize + self.opsize
        for i in range(total_size):
            for j in range(i):
                r = random.random()
                if (self.population[j].exceed_load == 0 and self.population[j + 1].exceed_load == 0) \
                        or r < pf:
                    if self.population[j].quality > self.population[j + 1].quality:
                        self.population[j], self.population[j + 1] = self.population[j + 1], self.population[j]
                elif self.population[j].exceed_load > self.population[j + 1].exceed_load[j + 1]:
                    self.population[j], self.population[j + 1] = self.population[j + 1], self.population[j]

    def maens(self):
        best_fsb_solution = self.initialize()
        total_size = self.psize + self.opsize
        counter = 0
        wite = 0
        old_best = Solution(None, None, np.inf, 0)
        while counter < self.Gmax:
            counter += 1
            wite += 1
            ptr = self.psize
            child = None
            while ptr < total_size:
                # randomly select two parents
                s1, s2 = self.random_select()

                # crossover
                sx = self.SBX(s1, s2)
                if sx.exceed_load == 0 and sx.quality < best_fsb_solution.quality:
                    best_fsb_solution = sx
                    wite = 0

                # add sx into population if not exsist
                if sx not in self.population[:ptr]:
                    child = sx

                # local search with probability
                r = random.random()
                if r < self.pls:
                    # do local search
                    sls, best_fsb_solution = self.lns_mut(sx, best_fsb_solution)
                    if sls not in self.population[:ptr]:
                        child = sls

                if child.quality > 0 and child != s1 and child != s2:
                    self.population[ptr] = child
                    ptr += 1

            # stochastic ranking
            self.stochastic_rank()

            if best_fsb_solution.quality < old_best.quality:
                old_best = best_fsb_solution
            print('MAENS: ', counter, ' ', best_fsb_solution.quality)

        # print('MAENS: ', counter, ' ', old_best.quality)
        return old_best

    def route_converter(self, dst: Solution, src: Solution):
        inst_task = self.info.tasks
        load = 0
        dst.task_seq[0] = 1
        dst.task_seq[1] = 0
        cnt_task, cnt_load = 0, 0
        # task_seq[i] == 0: 一条路径的结束点, 下一条路径的起始点;
        #                   相应load[i]对应该路径的load
        #             != 0: 任务编号
        for i in range(2, src.task_seq[0] + 1):
            if src.task_seq[i] == 0:
                cnt_task += 1
                dst.task_seq[cnt_task] = 0
                cnt_load += 1
                dst.loads[cnt_load] = load
                continue
            load += inst_task[src.task_seq[i]].demand
            cnt_task += 1
            dst.task_seq[cnt_task] = src.task_seq[i]
        dst.task_seq[0] = cnt_task
        dst.loads[0] = cnt_load