"""率定参数，using DEAP"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator
import random
from deap import tools

from src.calibrate.stat import statNse
from src.xaj.xaj import xaj


def cal_fitness(p_and_e, qobs, xaj_params, init_states):
    """统计预报误差等，计算模型fitness，也便于后面进行参数率定"""
    print("----------------------------------------一次径流模拟开始-------------------------------------------------")
    # 调用模型计算，得到输出
    simulated_flow = xaj(p_and_e, xaj_params, states=init_states)
    # 计算适应度，先以nse为例
    # 输出各个指标计算结果
    nses = statNse(qobs.reshape(qobs.shape[0], qobs.shape[1]),
                   simulated_flow.reshape(simulated_flow.shape[0], simulated_flow.shape[1]))
    nse = nses.mean()
    print("-----------------本次模拟纳什系数为：" + str(nse) + "------------------------")
    print("----------------------------------------一次径流模拟结束！-------------------------------------------------")
    print(" ")
    return nse


def evaluate(individual, x_input, y_true, init_states):
    # individual is the params of XAJ
    # we initialize the parameters in range [0,1] so here we denormalize the data to feasible range
    print(individual)
    param_ranges = OrderedDict({
        "B": [0.1, 0.4],
        "IM": [0.01, 0.04],
        "UM": [10, 20],
        "LM": [60, 90],
        "DM": [50, 90],
        "C": [0.1, 0.2],
        "SM": [10, 60],
        "EX": [1.0, 1.5],
        "KI": [0.1, 0.5],
        "KG": [0.1, 0.5],
        "CS": [0.1, 0.5],
        "CI": [0, 0.9],
        "CG": [0.9, 1],
    })
    norm = [(value[1] - value[0]) * individual[i] + value[0] for i, (key, value) in enumerate(param_ranges.items())]
    xaj_params = dict(zip(list(param_ranges.keys()), norm))
    return cal_fitness(x_input, y_true, xaj_params, init_states),  # this comma must exist


def calibrate_xaj(xaj_input, observed_output, init_states=None, param_num: int = 13,
                  run_counts: int = 40, pop_num: int = 50):
    """调用优化计算模型进行参数优选
    Parameters
    ----------
    xaj_input: the input data for XAJ
    observed_output: the "true" values, i.e. observations
    init_states: the initial states
    param_num: the number of parameters
    run_counts:  运行次数
    pop_num: the number of individuals in the population
    Returns
    ---------
    optimal_params
    """
    IND_SIZE = param_num
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, x_input=xaj_input, y_true=observed_output, init_states=init_states)
    pop = toolbox.population(n=pop_num)
    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

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
