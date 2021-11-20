import copy
import random

import numpy as np
from definition import Solution, Insert, Move

MAX_TASK_TAG_LENGTH = 500
MAX_TASK_SEG_LENGTH = 550
MAX_TASK_SEQ_LENGTH = 500
task_routes = np.zeros((101, MAX_TASK_SEG_LENGTH))

MAX_NSIZE = 10  # upper bound of n size
MAX_ENSSIZE = 100  # max ENS neighborhood size


def find_element_position(position, arr, e):
    position[0] = 0
    idx = np.argwhere(arr[1:arr[0] + 1], e)
    position[position[0]] = idx + 1
    position[0] += 1


def del_element(arr, k):
    if k < 1 or k > arr[0]:
        print('delete position error')
        return
    end = arr[0]
    arr[k:end] = arr[k + 1:end + 1]
    arr[end] = 0
    arr[0] -= 1


def del_element_by_e(arr, e):
    idx = np.argwhere(arr[1:arr[0] + 1], e)
    if idx < len(arr):
        arr = arr[:idx] + arr[idx + 1:]
    else:
        arr = arr[:idx]
    arr[0] -= 1


def add_element(arr, e, k):
    if k < 1 or k > arr[0] + 1:
        print('insert position error')
        return
    arr = np.insert(arr, k, e)


def assign_subroute(route1, k1, k2, route2):
    # assign route1[k1:k2] to route2
    length = k2 - k1 + 1
    route2[0] = length
    route2[1:length] = route1[k1:k2 + 1]


def join_routes(route1, route2):
    # for i in range(1, route2[0] + 1):
    #     route1[0] += 1
    #     route1[route1[0]] = route2[i]
    from_idx = route1[0]
    length = route2[0]
    to_idx = from_idx + length
    route1[from_idx:to_idx + 1] = route2[1:length + 1]


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


    def rand_scanning(self, inst_task, serve_mark):
        info = self.info
        serve_task_num = np.sum(serve_mark)
        task_num = info.edges_non_required + serve_task_num

        positions = np.zeros(serve_task_num + 1)
        nearest_task = np.zeros(serve_task_num + 1)

        sequence = np.zeros(serve_task_num + 1)
        sequence[0] = 1
        loads = np.zeros(serve_task_num + 1)
        unserved_task = np.zeros(serve_task_num + 1)
        serve_mark = np.zeros(task_num + 1)

        for i in range(1, task_num + 1):
            if not serve_mark[i]:
                continue
            unserved_task[0] += 1
            unserved_task[unserved_task[0]] = i

        load = 0
        trial = 0
        while trial < serve_task_num:
            curr_task = sequence[sequence[0]]
            candi_task = 0
            for i in range(1, unserved_task[0] + 1):
                if inst_task[unserved_task[i]].demand + load <= info.capacity:
                    candi_task[0] += 1
                    candi_task[candi_task[0]] = unserved_task[i]

            if candi_task[0] == 0:
                sequence[0] += 1
                sequence[sequence[0]] = 0
                loads[0] += 1
                loads[loads[0]] = load
                continue

            mindist = np.inf
            nearest_task[0] = 0

            for i in range(1, candi_task[0] + 1):
                curr_cost = info.min_dist[inst_task[curr_task].v, inst_task[candi_task[i]].u]
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
            load += inst_task[next_task].demand

            find_element_position(positions, unserved_task, next_task)
            del_element(unserved_task, positions[1])

        sequence[0] += 1
        sequence[sequence[0]] = 0

        loads[0] += 1
        loads[loads[0]] = load

        total_cost = self.get_task_seq_cost(sequence, inst_task)
        exceed_loads = self.get_exceed_loads(loads)
        return Solution(sequence, loads, total_cost, exceed_loads)


    def path_scanning(self, ps_ind: Solution, inst_tasks, serve_mark):
        # min_cost, NRE, NRA, NVeh, capacity, is the extern variables.
        info = self.info
        task_num = info.edges_total
        serve_task_num = np.sum(serve_mark[task_num+1:task_num])

        unserved_task = np.empty(MAX_TASK_TAG_LENGTH)
        candidate_task = np.empty(MAX_TASK_TAG_LENGTH)
        nearest_task = np.empty(MAX_TASK_TAG_LENGTH)
        nearest_isol_task = np.empty(MAX_TASK_TAG_LENGTH)
        nearest_inci_task = np.empty(MAX_TASK_TAG_LENGTH)
        sel_task = np.empty(MAX_TASK_TAG_LENGTH)

        position = np.empty(MAX_TASK_SEG_LENGTH)

        # depot_dist = np.empty(MAX_TASK_TAG_LENGTH)
        depot_dist = info.min_dist[info.depot, :].copy()
        # yeild = np.empty(MAX_TASK_TAG_LENGTH, dtype=np.float32)
        yeild = np.array([inst_tasks[i].demand / inst_tasks[i].cost for i in np.where(serve_mark[:task_num+1])[0]])

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
        #         curr_cost = info.min_dist[inst_tasks[curr_task].v, inst_tasks[candidate_task[i]].u]
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
            tmp_ind = self.path_scanning_with_rule(self.rules[i], self.eq_rules[i], inst_tasks,
                                                   serve_mark, unserved_task, candidate_task,
                                                   nearest_task, nearest_inci_task, nearest_isol_task)
            tmp_sol_list.append(tmp_ind)
        best_ind = min(tmp_sol_list, key=self.get_quality)

        self.ind_route_converter(ps_ind, best_ind, inst_tasks)
        ps_ind.quality = best_ind.quality
        ps_ind.exceed_load = 0


    def get_quality(self, solution: Solution):
        return solution.quality

    def path_scanning_with_rule(self, rule, eq_rule, inst_tasks,
                                serve_mark, unserved_task, candidate_task,
                                nearest_task, nearest_inci_task, nearest_isol_task, sel_task):
        info = self.info
        sequence = np.zeros(MAX_TASK_SEQ_LENGTH)
        loads = np.zeros(101)
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
                elif self.is_equal(inst_tasks(task), inst_tasks(best), load, eq_rule)
                    sel_task[0] += 1
                    sel_task[sel_task[0]] = task
            k = 1
            next_task = sel_task[k]

            trial += 1
            sequence[0] += 1
            sequence[sequence[0]] = next_task
            load += inst_tasks[next_task].demand

            # delete the served task in unserved task array
            del_element_by_e(unserved_task, next_task)
            if inst_tasks[next_task].inverse > 0:
                del_element_by_e(unserved_task, inst_tasks[next_task].inverse)

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


    def ind_route_converter(self, dst: Solution, src: Solution, inst_task):
        load = 0
        dst.task_seq = np.zeros(MAX_TASK_SEQ_LENGTH)
        dst.loads = np.zeros(50)
        dst.task_seq[0] = 1
        dst.task_seq[1] = 0

        task_pointer = 1
        load_pointer = 0
        for task in src.task_seq[2:src.task_seq[0] + 1]:
            if task == 0:
                task_pointer += 1
                # dst.task_seq[task_pointer] = 0
                load_pointer += 1
                dst.loads[load_pointer] = load
                load = 0
                continue
            load += inst_task[task].demand
            task_pointer += 1
            dst.task_seq[task_pointer] = task
        dst.task_seq[0] = task_pointer
        dst.loads[0] = load_pointer


    def get_task_seq_cost(self, task_seq, inst_tasks):
        total_cost = 0
        min_dist = self.info.min_dist
        for t in task_seq[1:task_seq[0] + 1]:
            total_cost += min_dist[inst_tasks[t].v, inst_tasks[t + 1].u] + inst_tasks[t].cost
        return total_cost

    def get_exceed_loads(self, route_seg_load):
        exceed_load = 0
        capacity = self.info.capacity
        for i in route_seg_load[1:route_seg_load[0] + 1]:
            exceed_load += max(0, route_seg_load[i] - capacity)
        return exceed_load

    def SBX(self, s1, s2, inst_tasks):
        """ Sequence Based Crossover Operator
        S1 -> {R11, R12}
        S2 -> {R21, R22}
        Sx -> {任意拼接}
        remove duplicate task in NewR and insert missing task to NewR -> offspring
        """
        info = self.info
        max_task_seq_length = info.edges_total
        sub_path1 = np.zeros(max_task_seq_length)
        sub_path2 = np.zeros(max_task_seq_length)
        routes1 = np.zeros((101, max_task_seq_length))
        routes2 = np.zeros((101, max_task_seq_length))

        """ define: candidate selection:
                (routeID, position)
        """
        candidate_list1, candidate_list2 = [], []

        position = np.zeros(101)
        xclds = np.zeros(101)
        left_tasks = np.zeros(max_task_seq_length)

        find_element_position(position, s1.task_seq, 0)
        routes1[0, 0] = position[0] - 1
        for i in range(1, position[0]):
            assign_subroute(s1.task_seq, position[i], position[i + 1], routes1[i])

        find_element_position(position, s2.task_seq, 0)
        routes2[0, 0] = position[0] - 1
        for i in range(1, position[0]):
            assign_subroute(s2.task_seq, position[i], position[i + 1], routes2[i])

        xclds[:len(s1.loads)] = s1.loads.copy()

        for i in range(1, routes1[0, 0] + 1):
            for j in range(2, routes1[i, 0]):
                candidate_list1.append((i, j))

        for i in range(1, routes2[0, 0] + 1):
            for j in range(2, routes2[i, 0]):
                candidate_list2.append((i, j))

        k1 = random.randrange(1, len(candidate_list1))
        k2 = random.randrange(1, len(candidate_list2))

        assign_subroute(routes1[candidate_list1[k1][0]], 1, candidate_list1[k1][1], sub_path1)
        assign_subroute(routes2[candidate_list2[k2][0]], candidate_list2[k2][1], routes2[candidate_list2[k2][0]][0],
                        sub_path2)
        assign_subroute(routes1[candidate_list1[k1][0]], candidate_list1[k1][0] + 1,
                        routes1[candidate_list1[k1][0]][0] - 1, left_tasks)

        # remove duplicated task for routes1
        checked = np.zeros(max_task_seq_length)
        for i in range(1, sub_path2[0]):
            # if checked[i]:
            #     continue
            for j in range(sub_path1[0], 1, -1):
                if sub_path1[j] == sub_path2[i] or sub_path1[j] == inst_tasks[sub_path2[i]].inverse:
                    del_element(sub_path1, j)
                    checked[i] = 1
                    break

        for i in range(1, sub_path2[0]):
            if checked[i]:
                continue
            for j in range(left_tasks[0], 0, -1):
                if left_tasks[j] == sub_path2[i] or left_tasks[j] == inst_tasks[sub_path2[i]].invers:
                    del_element(left_tasks, j)
                    checked[i] = 1
                    break

        for i in range(1, sub_path2[0]):
            if checked[i]:
                continue
            for j in range(1, routes1[0, 0] + 1):
                if j == candidate_list1[k1][0]:
                    continue
                for k in range(routes1[j, 0], 1, -1):
                    if routes1[j, k] == sub_path2[i] or routes1[j, k] == inst_tasks[sub_path2[i].inverse]:
                        del_element(routes1[j], k)
                        xclds[j] -= inst_tasks[sub_path2[i]].demand
                        checked[i] = 1

                if checked[i]:
                    break

        join_routes(sub_path1, sub_path2)
        routes1[candidate_list1[k1][0]:] = sub_path1
        xclds[candidate_list1[k1][0]] = 0
        for i in range(2, routes1[candidate_list1[k1][0]]):
            xclds[candidate_list1[k1][0]] += inst_tasks[routes1[candidate_list1[k1][0]], i].demand

        left_tasks_cnt = left_tasks[0]

        # insert missing tasks
        candidate_insertions = []
        pare_to_set_insertions = []
        curr_route = 0

        for j in range(1, routes1[0, 0] + 1):
            if routes1[j, 0] == 2:
                continue
            curr_route += 1

        out = np.zeros(6000)
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
                    insert_cost = info.min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[curr_task].u] \
                                  + info.min_dist[inst_tasks[curr_task].v, inst_tasks[routes1[j, k]].u] \
                                  - info.min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[routes1[j, k]].u]
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

                    insert_cost = info.min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[curr_task].u] \
                                  + info.min_dist[inst_tasks[w].v, inst_tasks[routes1[j, k].u]] \
                                  - info.min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[routes1[j, k]].u]
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
            insert_cost = info.min_dist[info.depot, inst_tasks[curr_task].u] \
                          + info.min_dist[inst_tasks[curr_task].v, info.depot]
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
                add_element(routes1[best_insertion.routeID], best_insertion.task, best_insertion.position)
                xclds[best_insertion.routeID] += inst_tasks[best_insertion.task].demand

        # transfer routes1 to sequence
        sequence = np.zeros(info.edges_required + 1)
        sequence[0] = 1
        for i in range(1, routes1[0, 0] + 1):
            if routes1[i, 0] == 2:
                continue
            sequence[0] -= 1
            join_routes(sequence, routes1[i])

        loads = xclds.copy()
        for i in range(loads[0], 0, -1):
            if loads[i] == 0:
                del_element(loads, i)

        quality = self.get_task_seq_cost(sequence, inst_tasks)
        exceed_load = self.get_exceed_loads(loads)
        return Solution(sequence, loads, quality, exceed_load)

    def lns_mut(self, s: Solution, best_fsb_solution: Solution, inst_tasks):
        info = self.info
        sls = copy.deepcopy(s)
        coef = best_fsb_solution.quality / info.capacity * (
                best_fsb_solution.quality / s.quality + s.exceed_load / info.capacity + 1.0)
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

            tmp_ind = copy.deepcopy(sls)
            # traditional operator
            self.lns(sls, coef, 1, inst_tasks)
            if sls.fitness < tmp_ind.fitness:
                imp = 1
            if count1 > 50 and sls.quality < tmp_ind.quality:
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
            self.lns(sls, coef, 2, inst_tasks)
            if sls.fitness < tmp_ind.fitness:
                imp = 1
            if count1 > 50 and sls.quality < tmp_ind.quality:
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
            self.lns(sls, coef, 1, inst_tasks)
            if sls.fitness < tmp_ind.fitness:
                imp = 1
            if count1 > 50 and sls.quality < tmp_ind.quality:
                break
            if sls.exceed_load == 0 and sls.quality < best_fsb_solution.quality:
                best_fsb_solution = copy.deepcopy(sls)
        return best_fsb_solution

    def get_fitness(self, move: Move):
        return move.fitness

    def lns(self, ind: Solution, coef, nsize, inst_tasks):
        info = self.info
        ind.calculate_fitness(coef)

        if nsize == 1:
            # traditional move
            next_move: Move = min([operation(ind) for operation in self.operations], key=self.get_fitness)

            orig_ptr, targ_ptr = 0, 0
            seg_ptr1, seg_ptr2 = 0, 0
            for i in range(1, ind.task_seq[0]):
                if ind.task_seq[i] == 0:
                    if seg_ptr1 < next_move.orig_seg:
                        seg_ptr1 += 1
                    if seg_ptr2 < next_move.targ_seg:
                        seg_ptr2 += 1
                    if seg_ptr1 == next_move.targ_seg and orig_ptr == 0:
                        orig_ptr = i + next_move.orig_pos - 1
                    if seg_ptr2 == next_move.targ_seg and targ_ptr == 0:
                        targ_ptr = i + next_move.targ_pos - 1

                if orig_ptr != 0 and targ_ptr != 0:
                    break

            if next_move.type == 1:
                # single insertion
                del_element(ind.task_seq, orig_ptr)
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
                    add_element(ind.task_seq, next_move.task1, targ_ptr)
                    ind.loads[next_move.targ_seg] += inst_tasks[next_move.task1].demand
            elif next_move.type == 2:
                # doubel insertion
                del_element(ind.task_seq, orig_ptr + 1)
                del_element(ind.task_seq, orig_ptr)
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
                    add_element(ind.task_seq, next_move.task2, targ_ptr)
                    add_element(ind.task_seq, next_move.task1, targ_ptr)
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
                    del_element(ind.task_seq, orig_ptr + 1)
                else:
                    del_element(ind.task_seq, orig_ptr)
                del_element(ind.loads, next_move.orig_seg)
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

            candidate_combs = np.zeros((MAX_NSIZE + 1, MAX_ENSSIZE + 1))
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

            lns_routes = np.zeros(MAX_NSIZE + 1)
            for i in range(maxcount):
                lns_routes = candidate_combs[i].copy()

                sel_total_load = 0
                for j in range(1, lns_routes[0] + 1):
                    sel_total_load += ind.loads[lns_routes[j]]

                if sel_total_load > nsize * info.capacity:
                    continue

                serve_mark = np.zeros(MAX_TASK_TAG_LENGTH)
                serve_mark[0] = info.edges_required
                for j in range(1, lns_routes[0] + 1):
                    for k in range(2, task_routes[lns_routes[j, 0]]):
                        serve_mark[task_routes[lns_routes[j, k]]] = 1
                        serve_mark[inst_tasks[task_routes[lns_routes[j], k]].inverse] = 1

                self.path_scanning(inst_task, serve_mark)


    def single_insertion(self, solution: Solution):
        """
        Move operator: Single Insertion
         A task is removed from its current position
         and re-inserted into another position of the current solution or a new empty route.

         If the selected task belongs to an edge task, both its directions will
         be considered when inserting the task into the “target position.”

        :param solution: parent
        :return: child generated by Single Insertion
        """
        info = self.info
        new_solution = copy.deepcopy(solution)

        routes = new_solution.routes
        route_idx = random.randrange(0, len(routes))
        selected_route = routes[route_idx]

        task_idx = random.randrange(0, len(selected_route))
        u, v = selected_route[task_idx]
        task = info.tasks[(u, v)]

        # calculate changed selected arc costs
        prev_tail = selected_route[task_idx - 1][1] if task_idx != 0 else info.depot
        next_head = selected_route[task_idx + 1][0] if task_idx != len(selected_route) - 1 else info.depot
        cost_diff = info.min_dist[prev_tail, next_head] \
                    - info.min_dist[prev_tail, u] - info.min_dist[v, next_head] \
                    - task.cost

        new_solution.costs[route_idx] += cost_diff
        new_solution.total_cost += cost_diff
        new_solution.loads[route_idx] -= task.demand
        selected_task = selected_route.pop(task_idx)

        # get inserted index
        routes.append([])
        insert_route_idx = random.randrange(0, len(routes))
        insert_route = routes[insert_route_idx]
        insert_position = random.randint(0, len(insert_route))  # start <= N <= end

        # calculate changed inserted arc costs
        prev_tail = insert_route[insert_position - 1][1] if insert_position != 0 else info.depot
        next_head = insert_route[insert_position][0] if insert_position != len(insert_route) else info.depot

        cost_diff = info.min_dist[prev_tail, u] + info.min_dist[v, next_head] \
                    + task.cost - info.min_dist[prev_tail, next_head]
        reversed_cost_diff = info.min_dist[prev_tail, v] + info.min_dist[u, next_head] + task.cost - info.min_dist[
            prev_tail, next_head]  # (v, u)
        if reversed_cost_diff < cost_diff:
            selected_task = (v, u)
            cost_diff = reversed_cost_diff

        if not insert_route:  # means a new arc
            new_solution.costs.append(cost_diff)
            new_solution.loads.append(task.demand)
        else:
            del routes[-1]
            new_solution.costs[insert_route_idx] += cost_diff
            new_solution.loads[insert_route_idx] += task.demand
        new_solution.total_cost += cost_diff

        insert_route.insert(insert_position, selected_task)
        new_solution.validate()
        return new_solution

    def double_insertion(self, solution):
        info = self.info
        new_solution = copy.deepcopy(solution)

        routes = new_solution.routes
        route_idx = random.randrange(0, len(routes))
        while len(routes[route_idx]) < 2:
            route_idx = random.randrange(0, len(routes))

        selected_route = routes[route_idx]
        task_idx = random.randrange(0, len(selected_route) - 1)

        u1, v1 = selected_route[task_idx]
        u2, v2 = selected_route[task_idx + 1]
        task1 = info.tasks[(u1, v1)]
        task2 = info.tasks[(u2, v2)]

        # calculate changed selected arc costs
        prev_tail = selected_route[task_idx - 1][1] if task_idx != 0 else info.depot
        next_head = selected_route[task_idx + 2][0] if task_idx != len(selected_route) - 2 else info.depot

        cost_diff = info.min_dist[prev_tail, next_head] - info.min_dist[prev_tail, u1] \
                    - task1.cost - info.min_dist[v1, u2] \
                    - task2.cost - info.min_dist[v2, next_head]
        new_solution.costs[route_idx] += cost_diff
        new_solution.total_cost += cost_diff
        new_solution.loads[route_idx] -= task1.demand + task2.demand

        selected_task1 = selected_route.pop(task_idx)
        selected_task2 = selected_route.pop(task_idx)

        # get inserted index
        routes.append([])
        insert_route_idx = random.randrange(0, len(routes))
        insert_route = routes[insert_route_idx]
        insert_position = random.randint(0, len(insert_route))

        # calculate changed inserted arc costs
        prev_tail = insert_route[insert_position - 1][1] if insert_position != 0 else info.depot
        next_head = insert_route[insert_position][0] if insert_position != len(insert_route) else info.depot

        cost_diff = info.min_dist[prev_tail, u1] \
                    + task1.cost + info.min_dist[v1, u2] \
                    + task2.cost + info.min_dist[v2, next_head] \
                    - info.min_dist[prev_tail, next_head]
        reversed_cost_diff = info.min_dist[prev_tail, v2] \
                             + task1.cost + info.min_dist[u2, v1] \
                             + task2.cost + info.min_dist[u1, next_head] \
                             - info.min_dist[prev_tail, next_head]
        if reversed_cost_diff < cost_diff:
            selected_task1 = (v2, u2)
            selected_task2 = (v1, u1)
            cost_diff = reversed_cost_diff

        if not insert_route:  # means a new arc
            new_solution.costs.append(cost_diff)
            new_solution.loads.append(task1.demand + task2.demand)
        else:
            del routes[-1]
            new_solution.costs[insert_route_idx] += cost_diff
            new_solution.loads[insert_route_idx] += task1.demand + task2.demand
        new_solution.total_cost += cost_diff
        insert_route.insert(insert_position, selected_task2)
        insert_route.insert(insert_position, selected_task1)

        new_solution.validate()
        return new_solution

    def swap(self, solution):
        new_solution = copy.deepcopy(solution)
        info = self.info
        routes = new_solution.routes

        # select first task
        route_idx_1 = random.randrange(0, len(routes))
        select_route_1 = routes[route_idx_1]
        task_idx_1 = random.randrange(0, len(select_route_1))
        u1, v1 = select_route_1[task_idx_1]
        task1 = info.tasks[(u1, v1)]
        prev_tail1 = select_route_1[task_idx_1 - 1][1] if task_idx_1 != 0 else info.depot
        next_head1 = select_route_1[task_idx_1 + 1][0] if task_idx_1 != len(select_route_1) - 1 else info.depot

        # select second task
        route_idx_2 = route_idx_1
        select_route_2 = None
        task_idx_2 = task_idx_1
        while route_idx_2 == route_idx_1 and task_idx_2 == task_idx_1:
            route_idx_2 = random.randrange(0, len(routes))
            select_route_2 = routes[route_idx_2]
            task_idx_2 = random.randrange(0, len(select_route_2))
        u2, v2 = select_route_2[task_idx_2]
        task2 = info.tasks[(u2, v2)]
        prev_tail2 = select_route_2[task_idx_2 - 1][1] if task_idx_2 != 0 else info.depot
        next_head2 = select_route_2[task_idx_2 + 1][0] if task_idx_2 != len(select_route_2) - 1 else info.depot

        selected_task1 = select_route_1.pop(task_idx_1)
        if route_idx_1 == route_idx_2 and task_idx_1 < task_idx_2:
            selected_task2 = select_route_2.pop(task_idx_2 - 1)
        else:
            selected_task2 = select_route_2.pop(task_idx_2)

        # calculate cost1 change
        reduced_cost_1 = info.min_dist[prev_tail1, u1] + task1.cost \
                         + info.min_dist[v1, next_head1]
        cost_diff_1 = info.min_dist[prev_tail1, u2] + task2.cost \
                      + info.min_dist[v2, next_head1] - reduced_cost_1
        reversed_cost_diff_1 = info.min_dist[prev_tail1, v2] + task2.cost + info.min_dist[
            u2, next_head1] - reduced_cost_1
        if reversed_cost_diff_1 < cost_diff_1:
            selected_task2 = (v2, u2)
            cost_diff_1 = reversed_cost_diff_1

        # insert task2 into route1
        new_solution.costs[route_idx_1] += cost_diff_1
        new_solution.total_cost += cost_diff_1
        new_solution.loads[route_idx_1] += (task2.demand - task1.demand)
        select_route_1.insert(task_idx_1, selected_task2)

        # calculate cost2 change
        reduced_cost_2 = info.min_dist[prev_tail2, u2] + task2.cost \
                         + info.min_dist[v2, next_head2]
        cost_diff_2 = info.min_dist[prev_tail2, u1] + task1.cost \
                      + info.min_dist[v1, next_head2] - reduced_cost_2
        reversed_cost_diff_2 = info.min_dist[prev_tail2, v1] + task1.cost \
                               + info.min_dist[u1, next_head2] - reduced_cost_2

        if reversed_cost_diff_2 < cost_diff_2:
            selected_task1 = (v1, u1)
            cost_diff_2 = reversed_cost_diff_2

        # insert task1 into route2
        new_solution.costs[route_idx_2] += cost_diff_2
        new_solution.total_cost += cost_diff_2
        new_solution.loads[route_idx_2] += (task1.demand - task2.demand)

        select_route_2.insert(task_idx_2, selected_task1)

        if route_idx_1 == route_idx_2:
            new_solution.total_cost = new_solution.quality(info)

        new_solution.validate()
        return new_solution

    def initialize(self, inst_tasks):
        trial = 0
        population = set()
        best_fsb_solution = Solution(None, None, np.inf, 0)
        while trial < self.ubtrial and len(population) < self.psize:
            trial += 1
            init_ind = self.rand_scanning(inst_tasks)
            if init_ind in population and trial != self.ubtrial:
                continue
            population.add(init_ind)
            if init_ind.exceed_load == 0 and init_ind.quality < best_fsb_solution:
                best_fsb_solution = init_ind
        self.psize = len(population)
        return best_fsb_solution

    def maens(self, inst_tasks):
        best_fsb_solution = self.initialize(inst_tasks)
        counter = 0
        wite = 0
        while counter < self.Gmax:
            counter += 1
            wite += 1

            ptr = self.psize
            child = None
            while ptr < self.info.edges_total:
                # randomly select two parents
                s1 = random.choice(self.population)
                s2 = random.choice(self.population)
                while s1 == s2:
                    s2 = random.choice(self.population)

                # crossover
                sx = self.SBX(s1, s2, inst_tasks)
                if sx.exceed_load == 0 and sx.quality < best_fsb_solution.quality:
                    best_fsb_solution = sx
                    wite = 0

                # add sx into population if not exsist
                if sx not in self.population:
                    child = sx

                # local search with probability
                r = random.random()
                if r < self.pls:
                    # do local search
                    sls = self.lns_mut()
