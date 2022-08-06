"""Calibrate XAJ model using DEAP"""
from deap import base, creator
import random
from deap import tools
import numpy as np
from hydromodel.utils.stat import statRmse
from hydromodel.models.gr4j import gr4j
from hydromodel.models.hymod import hymod
from hydromodel.models.xaj import xaj


def evaluate(individual, x_input, y_true, warmup_length, model):
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
    model
        model's name: "xaj", "xaj_mz", "gr4j", or "hymod"

    Returns
    -------
    float
        fitness
    """
    print("Calculate fitness:")
    # TODO: Now spotpy only support one list, and we only support one basin's calibration now
    params = np.array(individual).reshape(1, -1)
    if model == "xaj":
        sim = xaj(x_input, params, warmup_length=warmup_length)
    elif model == "xaj_mz":
        sim = xaj(x_input, params, warmup_length=warmup_length, route_method="MZ")
    elif model == "gr4j":
        sim = gr4j(x_input, params, warmup_length=warmup_length)
    elif model == "hymod":
        sim = hymod(x_input, params, warmup_length=warmup_length)
    else:
        raise NotImplementedError("We don't provide this model now")
    rmses = statRmse(y_true[warmup_length:, :, :], sim)
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


def calibrate_by_ga(
    input_data, observed_output, warmup_length=30, model="xaj", param_num=15, **ga_param
):
    """
    Use GA algorithm to find optimal parameters for hydrologic models

    Parameters
    ----------
    model
        model's name: "xaj", "xaj_mz", "gr4j", or "hymod"
    input_data
        the input data for model
    observed_output
        the "true" values, i.e. observations
    warmup_length
        the length of warmup period
    param_num
        the number of parameters for model
    ga_param
        run_counts: int = 40, running counts
        pop_num: int = 50, the number of individuals in the population
        cross_prob: float = 0.5, the probability with which two individuals are crossed
        mut_prob: float=0.5, the probability for mutating an individual

    Returns
    -------
    toolbox.population
        optimal_params
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute,
        n=param_num,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register(
        "evaluate",
        evaluate,
        x_input=input_data,
        y_true=observed_output,
        warmup_length=warmup_length,
        model=model,
    )

    toolbox.decorate("mate", checkBounds(MIN, MAX))
    toolbox.decorate("mutate", checkBounds(MIN, MAX))

    pop = toolbox.population(n=ga_param["pop_num"])
    # cxpb  is the probability with which two individuals are crossed
    # mutpb is the probability for mutating an individual
    cxpb, mutpb = ga_param["cross_prob"], ga_param["mut_prob"]

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(ga_param["run_counts"]):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
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
