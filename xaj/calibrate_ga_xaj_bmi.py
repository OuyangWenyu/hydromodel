"""Calibrate XAJ model using DEAP"""
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator
from deap import tools
from tqdm import tqdm

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.utils import hydro_constant, hydro_utils
from hydromodel.utils import stat
from hydromodel.utils.stat import statRmse
from hydromodel.visual.hydro_plot import plot_sim_and_obs, plot_train_iteration
from xaj.xaj_bmi import xajBmi


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
    # print("Calculate fitness:")
    # NOTE: Now only support one basin's calibration for once now
    params = np.array(individual).reshape(1, -1)
    if model["name"] in ["xaj", "xaj_mz"]:
        # xaj model's output include streamflow and evaporation now,
        # but now we only calibrate the model with streamflow
        model = xajBmi()
        model.initialize(os.path.relpath('runxaj.yaml'), params, x_input)
        while model.get_current_time() <= model.get_end_time('train'):
            model.update()
        sim = model.get_value("discharge")
        sim = np.expand_dims(sim, 0)
        sim = np.expand_dims(sim, 1)
        sim = np.transpose(sim, [2, 1, 0])
    else:
        raise NotImplementedError("We don't provide this model now")
    rmses = statRmse(y_true[warmup_length:, :, :], sim)
    rmse = rmses.mean(axis=0)
    # print(f"-----------------RMSE: {str(rmse)}------------------------")
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
        input_data, observed_output, deap_dir, warmup_length=30, model=None, ga_param=None
):
    """
    Use GA algorithm to find optimal parameters for hydrologic models

    Parameters
    ----------
    input_data
        the input data for model
    observed_output
        the "true" values, i.e. observations
    deap_dir
        the directory to save the results
    warmup_length
        the length of warmup period
    model
        the model setting
    ga_param
        random_seed: 1234
        run_counts: int = 40, running counts
        pop_num: int = 50, the number of individuals in the population
        cross_prob: float = 0.5, the probability with which two individuals are crossed
        mut_prob: float=0.5, the probability for mutating an individual

    Returns
    -------
    toolbox.population
        optimal_params
    """
    if model is None:
        model = {
            "name": "xaj_mz",
            "source_type": "sources",
            "source_book": "HF",
        }
    if ga_param is None:
        ga_param = {
            "random_seed": 1234,
            "run_counts": 5,
            "pop_num": 50,
            "cross_prob": 0.5,
            "mut_prob": 0.5,
            "save_freq": 1,
        }
    np.random.seed(ga_param["random_seed"])
    param_num = len(MODEL_PARAM_DICT[model["name"]]["param_name"])
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

    # save the best individual
    halloffame = tools.HallOfFame(maxsize=1)
    logbook = tools.Logbook()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Evaluate the entire population for the first time
    print("Initiliazing population...")
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    halloffame.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    cp = dict(
        population=pop,
        generation=0,
        halloffame=halloffame,
        logbook=logbook,
        rndstate=random.getstate(),
    )
    with open(os.path.join(deap_dir, "epoch0.pkl"), "wb") as cp_file:
        pickle.dump(cp, cp_file)

    for gen in tqdm(range(ga_param["run_counts"]), desc="GA calibrating"):

        print(f"Generation {gen} started...")
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
        for ind, fit in tqdm(
                zip(invalid_ind, fitnesses),
                desc=f"{str(gen + 1)} generation fitness calculating",
        ):
            ind.fitness.values = fit

        halloffame.update(offspring)
        record = stats.compile(offspring)
        # +1 means start from 1, 0 means initial generation
        logbook.record(gen=gen + 1, evals=len(invalid_ind), **record)
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        print(
            f"Best individual of {str(gen + 1)}"
            + f" generation is: {halloffame[0]}, {halloffame[0].fitness.values}"
        )
        if gen % ga_param["save_freq"] == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(
                population=pop,
                generation=gen + 1,
                halloffame=halloffame,
                logbook=logbook,
                rndstate=random.getstate(),
            )

            with open(
                    os.path.join(deap_dir, f"epoch{str(gen + 1)}.pkl"), "wb"
            ) as cp_file:
                pickle.dump(cp, cp_file)
            print(f"Files of generation {gen} saved.")
    top10 = tools.selBest(pop, k=10)
    return pop


def show_ga_result(
        deap_dir,
        warmup_length,
        basin_id,
        the_data,
        the_period,
        basin_area,
        model_info,
        result_unit="mm/day",
        train_mode=True,
):
    """
    show the result of GA
    """
    # https://stackoverflow.com/questions/61065222/python-deap-and-multiprocessing-on-windows-attributeerror
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    with open(os.path.join(deap_dir, "epoch2.pkl"), "rb") as cp_file:
        cp = pickle.load(cp_file)
    pop = cp["population"]
    logbook = cp["logbook"]
    halloffame = cp["halloffame"]
    print(f"Best individual is: {halloffame[0]}, {halloffame[0].fitness.values}")
    train_test_flag = "train" if train_mode else "test"

    model = xajBmi()
    model.initialize("runxaj.yaml", np.array(list(halloffame[0])).reshape(1, -1), the_data[:, :, 0:2])
    while model.get_current_time() <= model.get_end_time('train'):
        model.update()
    best_simulation = model.get_value("discharge")

    convert_unit_sim = hydro_constant.convert_unit(
        np.array(best_simulation).reshape(1, -1),
        # best_simulation,
        result_unit,
        hydro_constant.unit["streamflow"],
        basin_area=basin_area,
    )
    convert_unit_obs = hydro_constant.convert_unit(
        np.array(the_data[warmup_length:, :, -1:]).reshape(1, -1),
        result_unit,
        hydro_constant.unit["streamflow"],
        basin_area=basin_area,
    )
    # save calibrated results of calibration period
    the_result_file = os.path.join(
        deap_dir,
        f"{train_test_flag}_qsim_" + model_info["name"] + "_" + str(basin_id) + ".csv",
    )
    pd.DataFrame(convert_unit_sim.reshape(-1, 1)).to_csv(
        the_result_file,
        sep=",",
        index=False,
        header=False,
    )
    # calculation rmse、nashsutcliffe and bias for training period
    stat_error = stat.statError(
        convert_unit_obs,
        convert_unit_sim,
    )
    print(f"{train_test_flag}ing metrics:", basin_id, stat_error)
    hydro_utils.serialize_json_np(
        stat_error, os.path.join(deap_dir, f"{train_test_flag}_metrics.json")
    )
    t_range = pd.to_datetime(the_period[warmup_length:]).values.astype("datetime64[D]")
    save_fig = os.path.join(deap_dir, f"{train_test_flag}_results.png")
    if train_mode:
        save_param_file = os.path.join(deap_dir, basin_id + "_calibrate_params.txt")
        pd.DataFrame(list(halloffame[0])).to_csv(
            save_param_file, sep=",", index=False, header=True
        )
        fit_mins = logbook.select("min")
        plot_train_iteration(fit_mins, os.path.join(deap_dir, "train_iteration.png"))
    plot_sim_and_obs(
        t_range,
        convert_unit_sim.flatten(),
        convert_unit_obs.flatten(),
        save_fig,
    )


if __name__ == "__main__":
    data_dir = os.path.join(
        definitions.ROOT_DIR,
        "hydromodel",
        "example",
        "exp004",
    )
    deap_dir = os.path.join(
        data_dir,
        "Dec25_16-33-56_LAPTOP-DNQOPPMS_fold0_HFsources",
        "60668",
    )
    train_data_info_file = os.path.join(data_dir, "data_info_fold0_train.json")
    train_data_file = os.path.join(data_dir, "basins_lump_p_pe_q_fold0_train.npy")
    data_train = hydro_utils.unserialize_numpy(train_data_file)
    data_info_train = hydro_utils.unserialize_json_ordered(train_data_info_file)
    model_info = {
        "name": "xaj_mz",
        "source_type": "sources",
        "source_book": "HF",
    }
    train_period = data_info_train["time"]
    basin_area = data_info_train["area"][0]

    show_ga_result(
        deap_dir,
        365,
        "60668",
        data_train[:, 0:1, :],
        train_period,
        basin_area,
        model_info,
        result_unit="mm/day",
    )
