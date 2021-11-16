import os
import random
import time

from CARP_info import *
from v1_maen.CARP_ai import MAENS


# from CARP_algorithm import CARPAlgorithm as MAEN

class CarpEngine:
    def __init__(self, filename, termination, seed, population_size=100, test=False):
        self.info = Info(filename)
        self.termination = termination
        self.seed = seed
        self.population_size = population_size
        self.test = test

        random.seed(seed)
        np.random.seed(seed)

    def output(self, solution):
        def format_solution(s):
            s_print = []
            for p in s:
                # s_print.append(0)
                s_print.extend(p)
                # s_print.append(0)
            return s_print

        routes = solution.routes
        print("s", (",".join(str(d) for d in format_solution(routes))).replace(" ", ""))
        print("q", solution.total_cost)

    def solve(self):
        avg_time = 0
        total_time = 0
        remain_time = self.termination - 2
        start_time = time.perf_counter()

        solver = MAENS(self.info, self.population_size)
        remain_time -= (time.perf_counter() - start_time)

        iter_count = 0
        best = None
        while remain_time > 2 * avg_time:
            iter_start = time.perf_counter()
            best = solver.run()
            iter_end = time.perf_counter()
            iter_duration = iter_end - iter_start
            iter_count += 1
            total_time += iter_duration
            avg_time = 0.6 * avg_time + 0.4 * iter_duration
            remain_time -= iter_duration
            if self.test:
                print(
                    'pid: {} \tsample: {} \titer {} \t\tpopulation: {} \ttime: {:5.3} s \tavg: {:5.3} s \tremain: {:.3f} s \tcost: {}'.format(
                        os.getpid(),
                        self.info.name.split('\\')[-1],
                        iter_count,
                        len(solver.population),
                        iter_duration,
                        avg_time,
                        remain_time,
                        best.total_cost)
                )
        self.output(best)
        return int(best.total_cost)


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        sys.argv = ['CARP_solver.py', '../CARP_samples/gdb1.dat', '-t', '10', '-s', '1']

    filename, termination, seed = [sys.argv[i] for i in range(len(sys.argv)) if i % 2 == 1]
    termination = int(termination)
    seed = int(seed)

    engine = CarpEngine(filename, termination, seed, test=True)
    engine.solve()
