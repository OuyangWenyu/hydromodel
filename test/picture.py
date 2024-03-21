from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from numpy import *
import matplotlib.dates as mdates
import sys
from pathlib import Path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
# from hydromodel.utils import hydro_constant

time = pd.read_excel('D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/DMCA.xlsx')
time['starttime'] = pd.to_datetime(time['starttime'], format='%d/%m/%Y %H:%M')
time['endtime'] = pd.to_datetime(time['endtime'], format='%d/%m/%Y %H:%M')
sim = pd.read_excel('D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/picture.xlsx')
sim['date'] = pd.to_datetime(sim['date'], format='%d/%m/%Y %H:%M')
for i  in range(len(time)):
    start_time = time['starttime'][i]
    end_time = time['endtime'][i]
    start_num = np.where(sim['date'] == start_time)[0]
    end_num = np.where(sim['date'] == end_time)[0]
    # date = pd.date_range(start_time, end_time, freq='H')
    start_num = int(start_num)
    end_num = int(end_num)
    date =sim['date'][start_num:end_num]
    sim_xaj = sim['sim_xaj'][start_num:end_num]
    sim_dhf = sim['sim_dhf'][start_num:end_num]
    obs = sim['streamflow(m3/s)'][start_num:end_num]
    prcp = sim['prcp(mm/hour)'][start_num:end_num]
    fig = plt.figure(figsize=(9,6),dpi=500)
    ax = fig.subplots()
    ax.plot(
        date,
        sim_xaj,
        color="blue",
        linestyle="-",
        linewidth=1,
        label="Simulation_xaj",
    )
    ax.plot(
        date,
        sim_dhf,
        color="green",
        linestyle="-",
        linewidth=1,
        label="Simulation_dhf",
    )
    ax.plot(
        date,
        obs,
        # "r.",
        color="black",
        linestyle="-",
        linewidth=1,
        label="Observation",
    )
    ylim = np.max(np.vstack((obs, sim_xaj)))
    print(start_time)
    ax.set_ylim(0, ylim*1.3) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    xlabel="Date(∆t=1hour)"
    ylabel="Streamflow(m^3/s)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc="upper right")
    # sim_xaj = np.array(sim_xaj)
    # obs = np.array(obs)
    # numerator = 0
    # denominator = 0
    # meangauge = 0
    # count = 0
    # for h in range(len(obs)):
    #     if (obs[h]>=0):
    #         numerator+=pow(abs(sim_xaj[h])-obs[h],2)
    #         meangauge+=obs[h]
    #         count+=1
    # meangauge=meangauge/count
    # for m in range(len(obs)):
    #     if (obs[m]>=0):
    #         denominator+=pow(obs[m]-meangauge,2)
    # NSE= 1-numerator/denominator
    # plt.text(0.9, 0.6, 'NSE=%.2f' % NSE, 
    #      horizontalalignment='center',  
    #      verticalalignment='center',
    #      transform = ax.transAxes,
    #      fontsize=10)

    ax2 = ax.twinx()
    ax2.bar(date,prcp, label='Precipitation', color='royalblue',alpha=0.9,width=0.05)
    ax2.set_ylabel('Precipitation(mm)')
    plt.yticks(fontproperties = 'Times New Roman', size = 10)
    prcp_max = np.max(prcp)
    ax2.set_ylim(0, prcp_max*4)
    ax2.invert_yaxis()  #y轴反向
    ax2.legend(loc='upper left')
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    save_fig = os.path.join('D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/plot', "results"+str(i)+".png")
    plt.savefig(save_fig, bbox_inches="tight")
    plt.close()


def NSE(obs,sim_xaj):
    numerator = 0
    denominator = 0
    meangauge = 0
    count = 0
    for i in range(len(obs)):
        if (obs[i]>=0):
            numerator+=pow(abs(sim_xaj[i])-obs[i],2)
            meangauge+=obs[i]
            count+=1
    meangauge=meangauge/count
    for i in range(len(obs)):
        if (obs[i]>=0):
            denominator+=pow(obs[i]-meangauge,2)
    NSE= 1-numerator/denominator