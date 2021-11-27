import os
import random
import time
import numpy as np

from CARP_info import Info
from CARP_ai import MAENS

class CarpEngine:
    def __init__(self, filename, termination, seed, test=False):
        self.info = Info(filename)
        self.termination = termination
        self.seed = seed
        self.test = test
        random.seed(seed)
        np.random.seed(seed)

    def output(self, solution):
        def format_solution(s):
            s_print = []
            s_print.append(0)
            for p in s:
                if p == 0:
                    s_print.append(0)
                    s_print.append(0)
                    continue
                t = self.info.tasks[p]
                s_print.append((t.u, t.v))
            s_print.append(0)
            return s_print

        routes = solution.task_seq[2:solution.task_seq[0]]
        print("s", (",".join(str(d) for d in format_solution(routes))).replace(" ", ""))
        print("q", int(solution.quality))

    def solve(self):
        avg_time = 0
        total_time = 0
        remain_time = self.termination - 2
        start_time = time.perf_counter()

        solver = MAENS(self.info)
        remain_time -= (time.perf_counter() - start_time)

        iter_count = 0
        while remain_time > 2 * avg_time:
            iter_start = time.perf_counter()
            solver.solve()
            iter_end = time.perf_counter()
            iter_duration = iter_end - iter_start
            iter_count += 1
            total_time += iter_duration
            avg_time = 0.6 * avg_time + 0.4 * iter_duration
            remain_time -= iter_duration
            # if self.test:
            #     print(
            #         'pid: {} \tsample: {} \titer {} \t\tpopulation: {} \ttime: {:5.3} s \tavg: {:5.3} s \tremain: {:.3f} s \tcost: {}'.format(
            #             os.getpid(),
            #             self.info.name.split('\\')[-1],
            #             iter_count,
            #             len(solver.population),
            #             iter_duration,
            #             avg_time,
            #             remain_time,
            #             solver.best_fsb_solution.quality)
            #     )
        self.output(solver.best_fsb_solution)


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        sys.argv = ['CARP_solver.py', '../CARP_samples/egl-s1-A.dat', '-t', '30', '-s', '1']

    filename, termination, seed = [sys.argv[i] for i in range(len(sys.argv)) if i % 2 == 1]
    termination = int(termination)
    seed = int(seed) % 4294967296

    engine = CarpEngine(filename, termination, seed)
    engine.solve()
