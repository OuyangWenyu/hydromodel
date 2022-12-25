"""
Author: Wenyu Ouyang
Date: 2022-12-02 15:15:20
LastEditTime: 2022-12-02 15:17:08
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydro-model-xaj\hydromodel\visual\hydro_plot.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from matplotlib import pyplot as plt
import numpy as np

from hydromodel.utils import hydro_constant


def plot_rainfall_runoff(
    t,
    p,
    qs,
    fig_size=(8, 6),
    c_lst="rbkgcmy",
    leg_lst=None,
    dash_lines=None,
    title=None,
    xlabel=None,
    ylabel=None,
    linewidth=1,
):
    fig, ax = plt.subplots(figsize=fig_size)
    if dash_lines is not None:
        assert type(dash_lines) == list
    else:
        dash_lines = np.full(len(qs), False).tolist()
    for k in range(len(qs)):
        tt = t[k] if type(t) is list else t
        q = qs[k]
        leg_str = None
        if leg_lst is not None:
            leg_str = leg_lst[k]
        (line_i,) = ax.plot(tt, q, color=c_lst[k], label=leg_str, linewidth=linewidth)
        if dash_lines[k]:
            line_i.set_dashes([2, 2, 10, 2])

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
    # Create second axes, in order to get the bars from the top you can multiply by -1
    ax2 = ax.twinx()
    ax2.bar(tt, -p, color="b")

    # Now need to fix the axis labels
    max_pre = max(p)
    ax2.set_ylim(-max_pre * 5, 0)
    y2_ticks = np.arange(0, max_pre, 20)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax2.set_yticks(-1 * y2_ticks)
    ax2.set_yticklabels(y2_ticklabels, fontsize=16)
    # ax2.set_yticklabels([lab.get_text()[1:] for lab in ax2.get_yticklabels()])
    if title is not None:
        ax.set_title(title, loc="center", fontdict={"fontsize": 17})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("降水（mm/day）", fontsize=8, loc="top")
    # ax2.set_ylabel("precipitation (mm/day)", fontsize=12, loc='top')
    # https://github.com/matplotlib/matplotlib/issues/12318
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(bbox_to_anchor=(0.01, 0.9), loc="upper left", fontsize=16)
    ax.grid()


def plot_sim_and_obs(
    date,
    sim,
    obs,
    save_fig,
    xlabel="Date",
    ylabel="Streamflow(" + hydro_constant.unit["streamflow"] + ")",
):
    # matplotlib.use("Agg")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.plot(
        date,
        sim,
        color="black",
        linestyle="solid",
        label="Simulation",
    )
    ax.plot(
        date,
        obs,
        "r.",
        markersize=3,
        label="Observation",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_fig, bbox_inches="tight")
    # plt.cla()
    plt.close()


def plot_train_iteration(likelihood, save_fig):
    # matplotlib.use("Agg")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.plot(likelihood)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Iteration")
    plt.savefig(save_fig, bbox_inches="tight")
    # plt.cla()
    plt.close()
