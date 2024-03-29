import copy
import random

import numpy as np
from itertools import combinations

from CARP_utils import Solution, Insert, Move

Mtrial = 10
Mprob = 0.2
Gmax = 100
Mwite = 100

MAX_TASK_TAG_LENGTH = 500
MAX_TASK_SEG_LENGTH = 550
MAX_TASK_SEQ_LENGTH = 500

MAX_NSIZE = 10  # upper bound of n size
MAX_ENSSIZE = 100  # max ENS neighborhood size


def find_element_position(position, arr, e):
    idxs = np.where(arr[1:arr[0] + 1] == e)[0] + 1
    length = idxs.size
    position[1:length + 1] = idxs
    position[0] = length
    return position


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
    arr[arr[0]] = 0
    arr[0] -= 1
    return arr


def add_element(arr, e, k):
    if k < 1 or k > arr[0] + 1:
        print('insert position error')
        return
    arr[k + 1:arr[0] + 2] = arr[k:arr[0] + 1]
    arr[k] = e
    arr[0] += 1
    return arr


def assign_subroute(src, k1, k2, dst):
    # assign route1[k1:k2] to route2
    length = k2 - k1 + 1
    dst[0] = length
    dst[1:length + 1] = src[k1:k2 + 1]
    return dst


def join_routes(joint_arr, arr):
    length = arr[0]
    joint_arr[joint_arr[0] + 1:joint_arr[0] + length + 1] = arr[1:length + 1]
    joint_arr[0] += length
    return joint_arr


def ind_route_converter(src: Solution, inst_task):
    load = 0
    task_seq = np.zeros(MAX_TASK_SEQ_LENGTH, dtype=int)
    loads = np.zeros(50, dtype=int)
    task_seq[0] = 1
    task_seq[1] = 0
    for i in range(2, src.task_seq[0] + 1):
        if src.task_seq[i] == 0:
            task_seq[0] += 1
            task_seq[task_seq[0]] += 0
            loads[0] += 1
            loads[loads[0]] = load
            load = 0
            continue
        load += inst_task[src.task_seq[i]].demand
        task_seq[0] += 1
        task_seq[task_seq[0]] = src.task_seq[i]
    return Solution(task_seq=task_seq, loads=loads, quality=src.quality, exceed_load=0)


def chunk_task_seq(task_seq):
    task_routes = np.zeros((101, MAX_TASK_SEG_LENGTH), dtype=int)
    seg_start = np.where(task_seq[1:task_seq[0] + 1] == 0)[0] + 1
    for route_ptr in range(1, len(seg_start)):
        from_idx, to_idx = seg_start[route_ptr - 1], seg_start[route_ptr]
        length = to_idx - from_idx + 1
        task_routes[route_ptr][1:length + 1] = task_seq[from_idx:to_idx + 1]
        task_routes[route_ptr, 0] = length
    task_routes[0, 0] = len(seg_start) - 1
    return task_routes


class MAENS:
    def __init__(self, info, psize=20, ubtrial=50):
        # 任务信息初始化
        self.tasks = info.tasks
        self.req_edge_num = info.edges_required
        self.task_num = self.req_edge_num * 2
        self.capacity = info.capacity
        self.depot = info.depot
        self.min_dist = info.min_dist

        # 种群大小
        self.psize = psize
        self.population = []
        # 初始化寻找种群的最大迭代数
        self.ubtrial = ubtrial
        # 每次迭代生成的后代数量
        self.opsize = 6 * self.psize
        # 每回合生成的解的总数
        self.total_size = self.psize + self.opsize
        # Probability of carrying out local search (mutation)
        self.pls = 0.2
        # Number of routes involved in Merge-Split operator
        self.p = 2

        self.operations = [self.single_insertion, self.swap]

        from_head_to_depot = lambda x: self.min_dist[x, self.depot]
        self.rules = [
            lambda x, y, c: from_head_to_depot(x.v) > from_head_to_depot(y.v),
            lambda x, y, c: from_head_to_depot(x.v) < from_head_to_depot(y.v),
            lambda x, y, c: x.demand / x.cost > y.demand / y.cost,
            lambda x, y, c: x.demand / x.cost < y.demand / y.cost,
            lambda x, y, c: from_head_to_depot(x.v) > from_head_to_depot(y.v) if c < self.capacity / 2
            else from_head_to_depot(x.v) < from_head_to_depot(y.v)
        ]
        self.best_fsb_solution = self.initialize()

    def rand_scanning(self, serve_mark):
        inst_tasks = self.tasks
        capacity = self.capacity
        min_dist = self.min_dist
        serve_task_num = np.sum(serve_mark[self.req_edge_num + 1:self.task_num + 1])

        unserved_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        candi_tasks = np.empty(MAX_TASK_TAG_LENGTH + 1, dtype=int)
        nearest_task = np.empty(MAX_TASK_TAG_LENGTH + 1, dtype=int)
        cnt = 0
        for i in range(1, self.task_num + 1):
            if not serve_mark[i]:
                continue
            cnt += 1
            unserved_task[cnt] = i
        unserved_task[0] = cnt

        sequence = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
        sequence[0] = 1
        sequence[1] = 0
        loads = np.empty(MAX_TASK_SEG_LENGTH, dtype=int)
        loads[0] = 0

        load = 0
        trial = 0
        while trial < serve_task_num:
            if unserved_task[0] == 0:
                break
            curr_task = sequence[sequence[0]]
            counter = 0
            for i in range(1, unserved_task[0] + 1):
                if inst_tasks[unserved_task[i]].demand + load <= capacity:
                    counter += 1
                    candi_tasks[counter] = unserved_task[i]
            candi_tasks[0] = counter

            if candi_tasks[0] == 0:
                sequence[0] += 1
                sequence[sequence[0]] = 0
                loads[0] += 1
                loads[loads[0]] = load
                load = 0
                continue

            mindist = np.inf
            nearest_task[0] = 0

            # 随机获得下一个任务
            for i in range(1, candi_tasks[0] + 1):
                curr_candi_task = candi_tasks[i]
                curr_cost = min_dist[inst_tasks[curr_task].v, inst_tasks[curr_candi_task].u]
                if curr_cost < mindist:
                    mindist = curr_cost
                    nearest_task[0] = 1
                    nearest_task[nearest_task[0]] = curr_candi_task
                elif curr_cost == mindist:
                    nearest_task[0] += 1
                    nearest_task[nearest_task[0]] = curr_candi_task

            k = random.randrange(1, nearest_task[0] + 1)
            next_task = nearest_task[k]

            trial += 1
            sequence[0] += 1
            sequence[sequence[0]] = next_task
            load += inst_tasks[next_task].demand

            unserved_task = del_element_by_e(unserved_task, next_task)
            unserved_task = del_element_by_e(unserved_task, inst_tasks[next_task].inverse)

        sequence[0] += 1
        sequence[sequence[0]] = 0

        loads[0] += 1
        loads[loads[0]] = load

        total_cost = self.get_task_seq_cost(sequence, inst_tasks)
        exceed_loads = self.get_exceed_loads(loads)
        return Solution(sequence, loads, total_cost, exceed_loads)

    def path_scanning(self, serve_mark):
        inst_tasks = self.tasks
        min_dist = self.min_dist
        serve_task_num = np.sum(serve_mark[self.req_edge_num + 1:self.task_num + 1])

        unserved_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        candidate_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        nearest_task = np.empty(MAX_TASK_TAG_LENGTH, dtype=int)
        """Use five rules to obtain solution"""
        tmp_sol_list = []
        for rule in self.rules:
            sequence = np.empty(MAX_TASK_SEQ_LENGTH, dtype=int)
            loads = np.empty(MAX_TASK_SEG_LENGTH, dtype=int)
            sequence[0] = 1
            sequence[1] = 0
            loads[0] = 0

            cnt = 0
            for i in range(1, self.task_num + 1):
                if not serve_mark[i]:
                    continue
                cnt += 1
                unserved_task[cnt] = i
            unserved_task[0] = cnt

            load = 0
            trial = 0
            while trial < serve_task_num:
                curr_task = sequence[sequence[0]]

                cnt = 0
                for i in range(1, unserved_task[0] + 1):
                    if inst_tasks[unserved_task[i]].demand <= self.capacity - load:
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
                    curr_cost = min_dist[inst_tasks[curr_task].v, inst_tasks[candidate_task[i]].u]
                    if curr_cost < mindist:
                        mindist = curr_cost
                        nearest_task[0] = 1
                        nearest_task[1] = candidate_task[i]
                    elif curr_cost == mindist:
                        nearest_task[0] += 1
                        nearest_task[nearest_task[0]] = candidate_task[i]

                if nearest_task[0] == 0:
                    # no available task
                    break

                next_task = None
                for tsk in nearest_task[1:nearest_task[0] + 1]:
                    if next_task is None or self.is_better(inst_tasks[tsk], inst_tasks[next_task], load, rule):
                        next_task = tsk

                trial += 1
                sequence[0] += 1
                sequence[sequence[0]] = next_task
                load += inst_tasks[next_task].demand

                # delete the served task in unserved task array
                unserved_task = del_element_by_e(unserved_task, next_task)
                unserved_task = del_element_by_e(unserved_task, inst_tasks[next_task].inverse)

            sequence[0] += 1
            sequence[sequence[0]] = 0
            loads[0] += 1
            loads[loads[0]] = load
            quality = self.get_task_seq_cost(sequence, inst_tasks)
            # exceed_load = self.get_exceed_loads(loads)
            exceed_load = 0
            tmp_ind = Solution(task_seq=sequence, loads=loads, quality=quality, exceed_load=exceed_load)
            tmp_sol_list.append(tmp_ind)

        best_ind = min(tmp_sol_list, key=self.get_quality)
        return ind_route_converter(best_ind, inst_tasks)

    def get_quality(self, solution: Solution):
        return solution.quality

    def is_better(self, curr, prev, current_load, rule):
        return rule(curr, prev, current_load)

    def is_equal(self, curr, prev, current_load, rule):
        return rule(curr, prev, current_load)

    def get_task_seq_cost(self, task_seq, inst_tasks):
        total_cost = sum([self.min_dist[inst_tasks[task_seq[i]].v, inst_tasks[task_seq[i + 1]].u] \
                          + inst_tasks[task_seq[i]].cost for i in range(1, task_seq[0])])
        return total_cost

    def get_exceed_loads(self, route_seg_load):
        exceed_load = sum([max(route_seg_load[i] - self.capacity, 0) for i in range(1, route_seg_load[0] + 1)])
        return exceed_load

    def SBX(self, s1, s2):
        """ Sequence Based Crossover Operator
            S1 -> {R11, R12}
            S2 -> {R21, R22}
            Sx -> {任意拼接}
            remove duplicate task in NewR and insert missing task to NewR -> offspring
        """
        inst_tasks = self.tasks
        min_dist = self.min_dist
        capacity = self.capacity

        sub_path1 = np.zeros(MAX_TASK_SEQ_LENGTH, dtype=int)
        sub_path2 = np.zeros(MAX_TASK_SEQ_LENGTH, dtype=int)
        routes1 = np.zeros((MAX_TASK_SEG_LENGTH, MAX_TASK_SEQ_LENGTH), dtype=int)
        routes2 = np.zeros((MAX_TASK_SEG_LENGTH, MAX_TASK_SEQ_LENGTH), dtype=int)

        """ define: candidate selection:
                (routeID, position)
        """
        # starting index for each route
        position = np.zeros(MAX_TASK_SEG_LENGTH, dtype=int)
        # exceed loads for each route
        xclds = np.zeros(MAX_TASK_SEG_LENGTH, dtype=int)
        left_tasks = np.zeros(MAX_TASK_SEQ_LENGTH, dtype=int)

        position = find_element_position(position, s1.task_seq, 0)
        routes1[0, 0] = position[0] - 1
        for i in range(1, position[0]):
            routes1[i] = assign_subroute(s1.task_seq, position[i], position[i + 1], routes1[i])

        position = find_element_position(position, s2.task_seq, 0)
        routes2[0, 0] = position[0] - 1
        for i in range(1, position[0]):
            routes2[i] = assign_subroute(s2.task_seq, position[i], position[i + 1], routes2[i])

        xclds[:s1.loads[0] + 1] = s1.loads[:s1.loads[0] + 1]
        candidate_list1 = [(i, j) for i in range(1, routes1[0, 0] + 1) for j in range(2, routes1[i, 0])]
        candidate_list2 = [(i, j) for i in range(1, routes2[0, 0] + 1) for j in range(2, routes2[i, 0])]

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
        checked = np.zeros(sub_path2[0] + 1, dtype=int)
        for i in range(1, sub_path2[0]):
            for j in range(sub_path1[0], 1, -1):
                if sub_path1[j] == sub_path2[i] or sub_path1[j] == inst_tasks[sub_path2[i]].inverse:
                    sub_path1 = del_element(sub_path1, j)
                    checked[i] = 1
                    break
            if checked[i]:
                continue
            for j in range(left_tasks[0], 0, -1):
                if left_tasks[j] == sub_path2[i] or left_tasks[j] == inst_tasks[sub_path2[i]].inverse:
                    left_tasks = del_element(left_tasks, j)
                    checked[i] = 1
                    break
            if checked[i]:
                continue
            for j in range(1, routes1[0, 0] + 1):
                if j == sel_routeID_1:
                    continue
                for k in range(routes1[j, 0], 1, -1):
                    if routes1[j, k] == sub_path2[i] or routes1[j, k] == inst_tasks[sub_path2[i]].inverse:
                        routes1[j] = del_element(routes1[j], k)
                        xclds[j] -= inst_tasks[sub_path2[i]].demand
                        checked[i] = 1
                if checked[i]:
                    break

        sub_path1 = join_routes(sub_path1, sub_path2)
        routes1[sel_routeID_1] = sub_path1.copy()
        xclds[sel_routeID_1] = sum([inst_tasks[sub_path1[i]].demand for i in range(2, sub_path1[0])])

        left_tasks_cnt = left_tasks[0]
        # insert missing tasks
        candidate_insertions = [None] * 6000
        pare_to_set_insertions = [None] * 6000
        out = np.empty(6000, dtype=int)
        for n in range(1, left_tasks_cnt + 1):
            candidate_insert_cnt = 0
            pare_to_set_size = 0
            curr_task = 0
            for j in range(1, routes1[0, 0]):
                if routes1[j, 0] == 2:  # 空路径
                    continue
                curr_task = left_tasks[n]
                if xclds[j] > capacity:
                    ivload = inst_tasks[curr_task].demand
                elif xclds[j] + inst_tasks[curr_task].demand > capacity:
                    ivload = xclds[j] + inst_tasks[curr_task].demand - capacity
                else:
                    ivload = 0

                for k in range(2, routes1[j, 0] + 1):
                    candidate_insert_cnt += 1
                    insert_cost = min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[curr_task].u] \
                                  + min_dist[inst_tasks[curr_task].v, inst_tasks[routes1[j, k]].u] \
                                  - min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[routes1[j, k]].u]
                    # curr_task: 被选中的未被插入的task
                    # j: 要插入的routeID
                    # k: 被插入的route中对应的position
                    # cost: 插入之后增加的cost
                    # ivload: 插入之后增加的load
                    candidate_insertions[candidate_insert_cnt] = Insert(task=curr_task, routeID=j, position=k,
                                                                        cost=insert_cost, exceed_load=ivload)

                    out[0] = 0
                    add = True
                    for m in range(1, pare_to_set_size + 1):
                        cand_insert: Insert = candidate_insertions[candidate_insert_cnt]
                        if cand_insert.cost > pare_to_set_insertions[m].cost and \
                                cand_insert.exceed_load > pare_to_set_insertions[m].exceed_load:
                            add = False
                            break
                        elif cand_insert.cost < pare_to_set_insertions[m].cost and \
                                cand_insert.exceed_load < pare_to_set_insertions[m].exceed_load:
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
                    insert_cost = min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[w].u] \
                                  + min_dist[inst_tasks[w].v, inst_tasks[routes1[j, k]].u] \
                                  - min_dist[inst_tasks[routes1[j, k - 1]].v, inst_tasks[routes1[j, k]].u]
                    candidate_insertions[candidate_insert_cnt] = Insert(task=w, routeID=j, position=k, cost=insert_cost,
                                                                        exceed_load=ivload)

                    out[0] = 0
                    add = True
                    for m in range(1, pare_to_set_size + 1):
                        cand_insert: Insert = candidate_insertions[candidate_insert_cnt]
                        if cand_insert.cost > pare_to_set_insertions[m].cost and \
                                cand_insert.exceed_load > pare_to_set_insertions[m].exceed_load:
                            add = False
                            break
                        elif cand_insert.cost < pare_to_set_insertions[m].cost and \
                                cand_insert.exceed_load < pare_to_set_insertions[m].exceed_load:
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
            insert_cost = min_dist[self.depot, inst_tasks[curr_task].u] \
                          + min_dist[inst_tasks[curr_task].v, self.depot]
            candidate_insertions[candidate_insert_cnt] = Insert(task=curr_task, routeID=0, position=2, cost=insert_cost,
                                                                exceed_load=0)

            out[0] = 0
            add = True
            for m in range(1, pare_to_set_size + 1):
                cand_insert: Insert = candidate_insertions[candidate_insert_cnt]
                if cand_insert.cost > pare_to_set_insertions[m].cost and \
                        cand_insert.exceed_load > pare_to_set_insertions[m].exceed_load:
                    add = False
                    break
                elif cand_insert.cost < pare_to_set_insertions[m].cost and \
                        cand_insert.exceed_load < pare_to_set_insertions[m].exceed_load:
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
            best_insertion: Insert = pare_to_set_insertions[k]
            if best_insertion.routeID == 0:
                routes1[0, 0] += 1
                route_ptr = routes1[0, 0]
                routes1[route_ptr, 0] = 3
                routes1[route_ptr, 1] = 0
                routes1[route_ptr, 2] = best_insertion.task
                routes1[route_ptr, 3] = 0

                xclds[0] += 1
                xclds[xclds[0]] = inst_tasks[best_insertion.task].demand
            else:
                routes1[best_insertion.routeID] = add_element(routes1[best_insertion.routeID], best_insertion.task,
                                                              best_insertion.position)
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
        sls = copy.deepcopy(sx)
        coef = best_fsb_solution.quality / self.capacity * (
                best_fsb_solution.quality / sx.quality + sx.exceed_load / self.capacity + 1.0)
        sls.calculate_fitness(coef)

        count, count1 = 0, 0
        count_fsb, count_infsb = 0, 0

        nsize_arr = [1, 2, 1]
        for nsize in nsize_arr:
            imp = True  # improved
            count1 = 0
            while imp:
                count += 1
                count1 += 1
                imp = False

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

                prev_fitness = sls.fitness
                prev_quality = sls.quality
                # local search operator
                sls = self.lns(sls, coef, nsize)
                if sls.fitness < prev_fitness:
                    imp = True
                if count1 > 50 and sls.quality < prev_quality:
                    break
                if sls.exceed_load == 0 and sls.quality < best_fsb_solution.quality:
                    best_fsb_solution = copy.deepcopy(sls)
        return sls, best_fsb_solution

    def get_fitness(self, move: Move):
        return move.fitness

    def lns(self, ind: Solution, coef, nsize):
        inst_tasks = self.tasks
        ind.calculate_fitness(coef)

        if nsize == 1:
            # traditional move
            next_move: Move = min([move(ind, coef) for move in self.operations], key=self.get_fitness)
            # 将task_route转化为ind.task_seq
            seg_starts = np.ones(ind.loads[0] + 2, dtype=int)
            seg_starts[1:1 + ind.loads[0]] = np.where(ind.task_seq[1:ind.task_seq[0]] == 0)[0] + 1
            orig_ptr = seg_starts[next_move.orig_seg] + next_move.orig_pos - 1
            targ_ptr = seg_starts[next_move.targ_seg] + next_move.targ_pos - 1

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
            task_routes = chunk_task_seq(ind.task_seq)
            if task_routes[0, 0] < nsize:
                return

            # 计算可能产生的解(extended neighbor)的数量
            routes_index = np.arange(1, task_routes[0, 0] + 1, 1, dtype=int)
            combs = combinations(routes_index, nsize)
            candidate_combs = np.zeros((MAX_ENSSIZE + 1, MAX_NSIZE + 1), dtype=int)
            cnt = 0
            for tup in combs:
                cnt += 1
                if cnt >= MAX_ENSSIZE:
                    break
                candidate_combs[cnt, 0] = nsize
                for j in range(nsize):
                    candidate_combs[cnt, j + 1] = tup[j]
            maxcount = min(cnt, MAX_ENSSIZE)

            next_indi = Solution(None, None, np.inf, np.inf)
            next_indi.fitness = np.inf
            for i in range(1, maxcount + 1):
                lns_routes = candidate_combs[i].copy()

                sel_total_load = 0
                for j in range(1, lns_routes[0] + 1):
                    sel_total_load += ind.loads[lns_routes[j]]

                if sel_total_load > nsize * self.capacity:
                    continue

                serve_mark = np.zeros(MAX_TASK_TAG_LENGTH, dtype=int)
                for j in range(1, lns_routes[0] + 1):
                    routeID = lns_routes[j]
                    for k in range(2, task_routes[routeID, 0]):
                        serve_mark[task_routes[routeID, k]] = 1
                        serve_mark[inst_tasks[task_routes[routeID, k]].inverse] = 1

                tmp_indi = self.path_scanning(serve_mark)

                for j in range(1, task_routes[0, 0] + 1):
                    if j in lns_routes[1:lns_routes[0] + 1]:
                        continue
                    tmp_indi.task_seq[0] -= 1
                    tmp_indi.task_seq = join_routes(tmp_indi.task_seq, task_routes[j])
                    tmp_indi.loads[0] += 1
                    tmp_indi.loads[tmp_indi.loads[0]] = ind.loads[j]
                    tmp_indi.exceed_load += max(ind.loads[j] - self.capacity, 0)

                tmp_indi.quality = self.get_task_seq_cost(tmp_indi.task_seq, inst_tasks)
                # tmp_indi.exceed_load = self.get_exceed_loads(tmp_indi.loads)
                tmp_indi.calculate_fitness(coef)
                if tmp_indi.fitness < next_indi.fitness:
                    next_indi = copy.deepcopy(tmp_indi)

            if next_indi.fitness < ind.fitness:
                ind = copy.deepcopy(next_indi)
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
        inst_tasks = self.tasks
        min_dist = self.min_dist
        capacity = self.capacity

        """将individual 的task_seq转化为task_route格式
        task_routes[0, 0]: #routes
        task_routes[i, 0]: routes[i]的 #tasks
        task_toutes[i, j]: routes[i], task[j]在inst_task中对应的编号
        
        task_routes[i, 0] == 1: 
        
        ind.task_seq[i] == 0: 一条route的终点（下一条route起点）
                        != 0: route上的task在inst_task中的编号
        """
        task_routes = chunk_task_seq(ind.task_seq)

        best_move = Move(type=1)
        tmp_move = Move(type=1)
        # task_routes[0, 0]: segment总数
        for s1 in range(1, task_routes[0, 0] + 1):
            # orig_seg: 被remove的task所在的segment
            orig_seg = s1
            # task_routes[seg, 0]: segment中的路径条数
            for i in range(2, task_routes[s1, 0]):
                orig_pos = i
                for s2 in range(1, task_routes[0, 0] + 1):
                    if s2 == s1:
                        continue
                    # s2: 插入的segment
                    targ_seg = s2
                    for j in range(2, task_routes[s2, 0] + 1):
                        if inst_tasks[task_routes[s2, j - 1]].v == inst_tasks[task_routes[s2, j]].u:
                            continue
                        # targ_pos：插入的位置
                        targ_pos = j
                        # 计算新route的load
                        exceed_load = ind.exceed_load
                        exceed_load -= max(ind.loads[s1] - capacity, 0)
                        exceed_load -= max(ind.loads[s2] - capacity, 0)
                        exceed_load += max(ind.loads[s1] - inst_tasks[task_routes[s1, i]].demand - capacity, 0)
                        exceed_load += max(ind.loads[s2] + inst_tasks[task_routes[s1, i]].demand - capacity, 0)
                        exceed_load = max(exceed_load, 0)

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

                # insert as a new route
                targ_seg = task_routes[0, 0] + 1
                exceed_load = ind.exceed_load
                exceed_load -= max(ind.loads[s1] - capacity, 0)
                exceed_load += max(ind.loads[s1] - inst_tasks[task_routes[s1, i]].demand - capacity, 0)
                exceed_load = max(exceed_load, 0)

                task1 = task_routes[s1, i]
                quality = ind.quality \
                          + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                          + min_dist[self.depot, inst_tasks[task1].u] \
                          + min_dist[inst_tasks[task1].v, self.depot] \
                          - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                          - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u]
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
        return best_move

    def double_insertion(self, ind: Solution, coef):
        """double insertion:
        similar to single insertion.
        two consecutive tasks are moved instead of a single task
        :return: best move
        """
        inst_tasks = self.tasks
        min_dist = self.min_dist
        capacity = self.capacity
        task_routes = chunk_task_seq(ind.task_seq)

        best_move = Move(type=2)
        tmp_move = Move(type=2)
        for s1 in range(1, task_routes[0, 0] + 1):
            if task_routes[s1, 0] < 4:
                continue
            orig_seg = s1
            for i in range(2, task_routes[s1, 0] - 1):
                orig_pos = i
                task1_arr = [task_routes[s1, i], inst_tasks[task_routes[s1, i]].inverse]
                task2_arr = [task_routes[s1, i + 1], inst_tasks[task_routes[s1, i + 1]].inverse]
                for s2 in range(1, task_routes[0, 0] + 1):
                    if s2 == s1:
                        continue
                    targ_seg = s2

                    for j in range(2, task_routes[s2, 0] + 1):
                        if inst_tasks[task_routes[s2, j - 1]].v == inst_tasks[task_routes[s2, j]].u:
                            continue
                        targ_pos = j
                        exceed_load = ind.exceed_load
                        exceed_load -= max(ind.loads[s1] - capacity, 0)
                        exceed_load -= max(ind.loads[s2] - capacity, 0)
                        exceed_load += max(ind.loads[s1] \
                                           - inst_tasks[task_routes[s1, i]].demand \
                                           - inst_tasks[task_routes[s1, i + 1]].demand \
                                           - capacity, 0)
                        exceed_load += max(ind.loads[s2] \
                                           + inst_tasks[task_routes[s1, i]].demand \
                                           + inst_tasks[task_routes[s1, i + 1]].demand \
                                           - capacity, 0)
                        exceed_load = max(exceed_load, 0)

                        for task1 in task1_arr:
                            for task2 in task2_arr:
                                quality = ind.quality \
                                          + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[
                                    task_routes[s1, i + 2]].u] \
                                          - min_dist[
                                              inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                          - min_dist[
                                              inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                          - min_dist[inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[
                                    task_routes[s1, i + 2]].u] \
                                          + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                          + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                                          + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s2, j]].u] \
                                          - min_dist[
                                              inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u]
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

                # insert as a new route
                if task_routes[s1][0] == 4:
                    continue
                targ_seg = task_routes[0, 0] + 1
                exceed_load = ind.exceed_load
                exceed_load -= max(ind.loads[s1] - capacity, 0)
                exceed_load += max(ind.loads[s1] - inst_tasks[task_routes[s1, i]].demand \
                                   - inst_tasks[task_routes[s1, i + 1]].demand \
                                   - capacity, 0)
                exceed_load += max(inst_tasks[task_routes[s1, i]].demand \
                                   + inst_tasks[task_routes[s1, i + 1]].demand \
                                   - capacity, 0)
                exceed_load = max(exceed_load, 0)

                task1, task2 = task1_arr[0], task2_arr[0]
                quality = ind.quality \
                          + min_dist[
                              inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                          - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                          - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                          - min_dist[
                              inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                          + min_dist[self.depot, inst_tasks[task1].u] \
                          + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                          + min_dist[inst_tasks[task2].v, self.depot]
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

                task1, task2 = task1_arr[0], task2_arr[1]
                quality = ind.quality \
                          + min_dist[
                              inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                          - min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                          - min_dist[inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                          - min_dist[
                              inst_tasks[task_routes[s1, i + 1]].v, inst_tasks[task_routes[s1, i + 2]].u] \
                          + min_dist[self.depot, inst_tasks[task1].u] \
                          + min_dist[inst_tasks[task1].v, inst_tasks[task2].u] \
                          + min_dist[inst_tasks[task2].v, self.depot]
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
        return best_move

    def swap(self, ind: Solution, coef):
        inst_tasks = self.tasks
        min_dist = self.min_dist
        capacity = self.capacity

        task_routes = chunk_task_seq(ind.task_seq)

        best_move = Move(type=3)
        tmp_move = Move(type=3)
        for s1 in range(1, task_routes[0, 0] + 1):
            orig_seg = s1
            for i in range(2, task_routes[s1, 0]):
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
                        exceed_load_1_curr = max(ind.loads[s1]
                                                 - inst_tasks[task_routes[s1, i]].demand
                                                 + inst_tasks[task_routes[s2, j]].demand
                                                 - capacity, 0)
                        exceed_load_1_prev = max(ind.loads[s1] - capacity, 0)
                        exceed_load_2_curr = max(ind.loads[s2]
                                                 + inst_tasks[task_routes[s1, i]].demand
                                                 - inst_tasks[task_routes[s2, j]].demand
                                                 - capacity, 0)
                        exceed_load_2_prev = max(ind.loads[s2] - capacity, 0)
                        exceed_load += (exceed_load_1_curr - exceed_load_1_prev) \
                                       + (exceed_load_2_curr - exceed_load_2_prev)
                        exceed_load = max(exceed_load, 0)

                        cand_task1 = [task_routes[s1, i], inst_tasks[task_routes[s1, i]].inverse]
                        cand_task2 = [task_routes[s2, j], inst_tasks[task_routes[s2, j]].inverse]
                        for task1 in cand_task1:
                            for task2 in cand_task2:
                                quality = ind.quality \
                                          - min_dist[
                                              inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task_routes[s1, i]].u] \
                                          - min_dist[
                                              inst_tasks[task_routes[s1, i]].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                          + min_dist[inst_tasks[task_routes[s1, i - 1]].v, inst_tasks[task2].u] \
                                          + min_dist[inst_tasks[task2].v, inst_tasks[task_routes[s1, i + 1]].u] \
                                          - min_dist[
                                              inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task_routes[s2, j]].u] \
                                          - min_dist[
                                              inst_tasks[task_routes[s2, j]].v, inst_tasks[task_routes[s2, j + 1]].u] \
                                          + min_dist[inst_tasks[task_routes[s2, j - 1]].v, inst_tasks[task1].u] \
                                          + min_dist[inst_tasks[task1].v, inst_tasks[task_routes[s2, j + 1]].u]
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
        task_num = self.task_num
        population = []
        best_fsb_solution = Solution(None, None, np.inf, np.inf)
        # population.append(best_fsb_solution)
        while len(population) < self.psize:
            trial = 0
            while trial < Mtrial:
                serve_mark = np.zeros(MAX_TASK_TAG_LENGTH, dtype=int)
                serve_mark[1:task_num + 1] = 1
                init_ind = self.rand_scanning(serve_mark)
                if init_ind in population and trial != Mtrial:
                    continue
                elif trial == Mtrial:
                    break
                population.append(init_ind)
                if init_ind.exceed_load == 0 and init_ind.quality < best_fsb_solution.quality:
                    best_fsb_solution = init_ind
                break
        self.psize = len(population)
        self.opsize = 6 * self.psize
        self.total_size = self.psize + self.opsize
        self.population = population
        return best_fsb_solution

    def random_select(self):
        s1_idx = random.randrange(0, self.psize)
        s2_idx = random.randrange(0, self.psize)
        while s1_idx == s2_idx:
            s2_idx = random.randrange(0, self.psize)
        s1 = self.population[s1_idx]
        s2 = self.population[s2_idx]
        return s1, s2

    def stochastic_rank(self):
        pf = 0.45
        for i in range(self.total_size):
            for j in range(i):
                r = random.random()
                if (self.population[j].exceed_load == 0 and self.population[j + 1].exceed_load == 0) \
                        or r < pf:
                    if self.population[j].quality > self.population[j + 1].quality:
                        self.population[j], self.population[j + 1] = self.population[j + 1], self.population[j]
                elif self.population[j].exceed_load > self.population[j + 1].exceed_load:
                    self.population[j], self.population[j + 1] = self.population[j + 1], self.population[j]
        self.population = self.population[:self.psize]

    def maens(self):
        total_size = self.total_size
        counter = 0
        wite = 0
        old_best = Solution(None, None, np.inf, 0)
        while counter < 5:
            counter += 1
            wite += 1
            ptr = self.psize
            child = Solution(None, None, -1, -1)
            while ptr < total_size:
                # randomly select two parents
                s1, s2 = self.random_select()
                # crossover
                sx = self.SBX(s1, s2)
                if sx.exceed_load == 0 and sx.quality < self.best_fsb_solution.quality:
                    self.best_fsb_solution = sx
                    wite = 0

                # add sx into population if not exsist
                if sx not in self.population[:ptr]:
                    child = sx

                # local search with probability
                r = random.random()
                if r < self.pls:
                    # do local search
                    sls, self.best_fsb_solution = self.lns_mut(sx, self.best_fsb_solution)
                    if sls not in self.population[:ptr]:
                        child = sls

                if child.quality > 0 and child != s1 and child != s2:
                    self.population[ptr] = child
                    ptr += 1
            # stochastic ranking
            self.stochastic_rank()

            if self.best_fsb_solution.quality < old_best.quality:
                old_best = self.best_fsb_solution
            print('MAENS: ', counter, ' ', self.best_fsb_solution.quality)

    def solve(self):
        if len(self.population) == self.total_size:
            self.stochastic_rank()

        child = Solution(None, None, -1, -1)
        # randomly select two parents
        s1, s2 = self.random_select()
        # crossover
        sx = self.SBX(s1, s2)
        if sx.exceed_load == 0 and sx.quality < self.best_fsb_solution.quality:
            self.best_fsb_solution = sx
        # add sx into population if not exsist
        if sx not in self.population:
            child = sx
        # local search with probability
        r = random.random()
        if r < self.pls:
            # do local search
            sls, self.best_fsb_solution = self.lns_mut(sx, self.best_fsb_solution)
            if sls not in self.population:
                child = sls
        if child.quality > 0 and child != s1 and child != s2:
            self.population.append(child)
