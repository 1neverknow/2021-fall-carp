import random

from deap import base, creator, tools
from v1_ga.CARP_info import Info

"""Individual coding:
    all visited customers of a route (including several sub-routes)
    are coded into an individual in turn.
    For example:
        Sub-route 1: 0 - 5 - 3 - 2 - 0
        Sub-route 2: 0 - 7 - 1 - 6 - 9 - 0
        Sub-route 3: 0 - 8 - 4 - 0
    These routes are coded as: [5, 3, 7, 1, 6, 9, 8, 4]
"""


def ind2route(individual, info: Info):
    """ Individual decoding

    :param individual: An individual to be decoded.
    :param info: A problem instance information object (class: info)
    :return: A list of decoded sub-routes corresponding to the input individual.
    """


def print_route(route):
    """ Prints sub-routes information to scree

    :param route:  A route decoded by ind2route(individual, instance).
    :return: none
    """


def eval_carptw(individual, info):
    """ Evaluation
    takes one individual as argument and returns
    its fitness as a Python tuple object

    :param individual: An individual to be evaluated
    :param info: A problem instance information object (class: info)
    :return: tuple of one fitness value of the evaluated individual.
    """


def select(individuals, k):
    """ Roulette Wheel Selection
    selects k individuals from the input individuals using k spins of a roulette.
    The selection is made by looking only at the first objective of each individual.
    The list returned contains references to the input individuals.

    :param individuals: A list of individuals to select from
    :param k: The number of individuals to select.
    :return: A list of selected individuals.
    """


def mate(ind1, ind2):
    """ Partially Matched Crossover
    executes a partially matched crossover (PMX) on the input individuals.
    The two individuals are modified in place.
    This crossover expects sequence individuals of indexes,
    the result for any other type of individuals is unpredictable.

    """


def mutation(individual):
    """ Mutation: Inverse Operation
    inverses the attributes between two random points of the input individual and return the mutant.
    This mutation expects sequence individuals of indexes,
    the result for any other type of individuals is unpredictable.

    :param individual:
    :return:
    """


def fitness(individual):
    return individual.fitness.values[0]


def run(info, ind_size, pop_size, cx_pb, mut_pb, n_gen):
    """
    implements a genetic algorithm-based solution
    to CARP with time windows (CARPTW).

    :param info: Problem information object
    :param ind_size: Size of an individual.
    :param pop_size: Size of a population.
    :param cx_pb: Probability of crossover.
    :param mut_pb: Probability of mutation.
    :param n_gen: Maximum number of generations to terminate evolution.
    :return: best solution
    """
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Operator registering
    toolbox.register('evaluate', eval_carptw, info=info)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', mate)
    toolbox.register('mutate', mutation)
    pop = toolbox.population(n=pop_size)

    # Hold results by exporting to csv file
    csv_data = []
    print('Start of evolution')
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(f'  Evaluated {len(pop)} individuals')

    offspring = []
    # Begin evolution
    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
        # select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

    valid_population = [p for p in offspring if p.is_valid]
    return min(valid_population, key=fitness)
