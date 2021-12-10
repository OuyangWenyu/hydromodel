"""Calibrate XAJ model using DEAP"""
from deap import base, creator
import random
from deap import tools
import numpy as np
from hydromodel.calibrate.stat import statRmse
from hydromodel.models.xaj import xaj


def evaluate(individual, x_input, y_true, warmup_length):
    """
    Calculate fitness for optimization

    Parameters
    ----------
    individual
        individual is the params of XAJ (see details in xaj.py); we initialize all parameters in range [0,1]
    x_input
        input of XAJ
    y_true
        observation data; we use the part after warmup period
    warmup_length
        the length of warmup period

    Returns
    -------
    float
        fitness
    """
    print("Calculate fitness:")
    # TODO: Now spotpy only support one list, and we only support one basin's calibration now
    params = np.array(individual).reshape(-1, 1)
    simulated_flow = xaj(x_input, params, warmup_length=warmup_length)
    rmses = statRmse(y_true[warmup_length:, :, :], simulated_flow)
    rmse = rmses.mean(axis=0)
    print("-----------------RMSE：" + str(rmse) + "------------------------")
    return rmse


def checkBounds(min, max):
    """
    A decorator to set bounds for individuals in a population

    Parameters
    ----------
    min
        the lower bound of individuals
    max
        the upper bound of individuals

    Returns
    -------
    Function
        a wrapper for clipping data into a given bound
    """

    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring

        return wrapper

    return decorator


MIN = 0
MAX = 1


def calibrate_xaj_ga(xaj_input, observed_output, warmup_length=30, param_num: int = 14,
                     run_counts: int = 40, pop_num: int = 50):
    """
    Use GA algorithm to find optimal parameters for XAJ

    Parameters
    ----------
    xaj_input
        the input data for XAJ
    observed_output
        the "true" values, i.e. observations
    warmup_length
        the length of warmup period
    param_num
        the number of parameters is 14 for our XAJ implementation
    run_counts
        running counts
    pop_num
        the number of individuals in the population

    Returns
    -------
    toolbox.population
        optimal_params
    """
    IND_SIZE = param_num
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, x_input=xaj_input, y_true=observed_output, warmup_length=warmup_length)

    toolbox.decorate("mate", checkBounds(MIN, MAX))
    toolbox.decorate("mutate", checkBounds(MIN, MAX))

    pop = toolbox.population(n=pop_num)
    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.5

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(run_counts):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    return pop
