import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



time = pd.read_excel('D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/DMCA.xlsx')
time['starttime'] = pd.to_datetime(time['starttime'], format='%d/%m/%Y %H:%M')
time['endtime'] = pd.to_datetime(time['endtime'], format='%d/%m/%Y %H:%M')
sim = pd.read_excel('D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/picture.xlsx')
sim['date'] = pd.to_datetime(sim['date'], format='%d/%m/%Y %H:%M')
basin_area= 2814
assess_xaj_list = []
assess_dhf_list = []
W_bias_rela_xaj_list = []
Q_bias_rela_xaj_list = []
NSE_xaj_list = []
W_bias_rela_dhf_list = []
Q_bias_rela_dhf_list = []
NSE_dhf_list = []
W_obs_list = []
W_sim_xaj_list = []
W_sim_dhf_list = []
Q_max_obs_list = []
Q_max_sim_xaj_list = []
Q_max_sim_dhf_list = []
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
    obs = sim['streamflow(m3/s)'][start_num:end_num]
    sim_dhf = sim['sim_dhf'][start_num:end_num]
    sim_xaj = np.array(sim_xaj)
    obs = np.array(obs)
    sim_dhf = np.array(sim_dhf)
    numerator = 0
    denominator = 0
    meangauge = 0
    count = 0
    for h in range(len(obs)):
        if (obs[h]>=0):
            numerator+=pow(abs(sim_xaj[h])-obs[h],2)
            meangauge+=obs[h]
            count+=1
    meangauge=meangauge/count
    for m in range(len(obs)):
        if (obs[m]>=0):
            denominator+=pow(obs[m]-meangauge,2)
    NSE_xaj= 1-numerator/denominator
    W_obs=sum(obs)*3600*1000/basin_area/1000000
    W_sim_xaj = sum(sim_xaj) * 3600 * 1000 /basin_area/ 1000000
    W_bias_abs_xaj=W_sim_xaj-W_obs
    W_bias_rela_xaj = abs(W_sim_xaj-W_obs)/W_obs
    Q_max_obs=np.max(obs)
    Q_max_sim_xaj=np.max(sim_xaj)
    Q_bias_rela_xaj = abs(Q_max_sim_xaj-Q_max_obs)/Q_max_obs

    assess_xaj = [W_bias_rela_xaj,Q_bias_rela_xaj,NSE_xaj]
    numerator = 0
    denominator = 0
    meangauge = 0
    count = 0
    for h in range(len(obs)):
        if (obs[h]>=0):
            numerator+=pow(abs(sim_dhf[h])-obs[h],2)
            meangauge+=obs[h]
            count+=1
    meangauge=meangauge/count
    for m in range(len(obs)):
        if (obs[m]>=0):
            denominator+=pow(obs[m]-meangauge,2)
    NSE_dhf= 1-numerator/denominator
    W_sim_dhf = sum(sim_dhf) * 3600 * 1000 /basin_area/ 1000000
    W_bias_abs_dhf=W_sim_dhf-W_obs
    W_bias_rela_dhf = abs(W_sim_dhf-W_obs)/W_obs
    Q_max_obs=np.max(obs)
    Q_max_sim_dhf=np.max(sim_dhf)
    Q_bias_rela_dhf = abs(Q_max_sim_dhf-Q_max_obs)/Q_max_obs
    assess_dhf = [W_bias_rela_dhf,Q_bias_rela_dhf,NSE_dhf]
    W_bias_rela_xaj_list.append(W_bias_rela_xaj)
    Q_bias_rela_xaj_list.append(Q_bias_rela_xaj)
    NSE_xaj_list.append(NSE_xaj)
    W_bias_rela_dhf_list.append(W_bias_rela_dhf)
    Q_bias_rela_dhf_list.append(Q_bias_rela_dhf)
    NSE_dhf_list.append(NSE_dhf)
    W_obs_list.append (W_obs)
    W_sim_xaj_list.append(W_sim_xaj)
    W_sim_dhf_list.append(W_sim_dhf)
    Q_max_obs_list.append(Q_max_obs)
    Q_max_sim_xaj_list.append(Q_max_sim_xaj)
    Q_max_sim_dhf_list.append(Q_max_sim_dhf)
    assess_obs=[0,0,1]
    x = np.vstack((assess_obs,assess_xaj, assess_dhf))
    # 找出最后一列小于 0 的行的索引
    negative_indices = np.where(x[:, -1] < 0)[0]

    # 如果存在最后一列小于 0 的行，则对这些行进行归一化处理
    if negative_indices.any():  
        for idx in negative_indices:
            x[idx] = (x[idx]-min(x[:, -1]))/(max(x[:, -1])-min(x[:, -1]))
    dfmax = x
    diso_dm3 = np.zeros((3, 1))
    for i in range(2):
        dm3 = np.sqrt((dfmax[i+1, 0] - dfmax[0, 0])**2 + (dfmax[i+1, 1] - dfmax[0, 1])**2 + (dfmax[i+1, 2] - dfmax[0, 2])**2)
        diso_dm3[i, 0] = dm3
    
    assess_xaj_nor = diso_dm3[0,0]
    assess_dhf_nor = diso_dm3[1,0]
    assess_xaj_list.append(assess_xaj_nor)
    assess_dhf_list.append(assess_dhf_nor)
   
changci_df = pd.DataFrame({
                        'start_time':time['starttime'],
                        'end_time':time['endtime'],
                        'W_obs':W_obs_list,
                        'W_sim_xaj':W_sim_xaj_list,
                        'W_sim_dhf':W_sim_dhf_list,
                        'Q_max_obs':Q_max_obs_list,
                        'Q_max_sim_xaj':Q_max_sim_xaj_list,
                        'Q_max_sim_dhf':Q_max_sim_dhf_list,
                       'W_bias_rela_xaj':W_bias_rela_xaj_list,
                       'W_bias_rela_dhf':W_bias_rela_dhf_list,
                       'Q_bias_rela_xaj':Q_bias_rela_xaj_list,
                       'Q_bias_rela_dhf':Q_bias_rela_dhf_list,
                       'NSE_xaj':NSE_xaj_list,
                       'NSE_dhf':NSE_dhf_list,
                        'assess_xaj':assess_xaj_list,
                        'assess_dhf':assess_dhf_list
                    })
changci_df.to_excel('D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/changci.xlsx')
    # # 拼接 diso 矩阵
    # diso = np.concatenate((diso_dm1, diso_dm2, diso_dm3), axis=1)

    # # 绘制二维 diso 图
    # plt.figure()
    # plt.scatter(dfmax[:, 0], dfmax[:, 1], marker='o')

    # # 添加文本标注
    # plt.text(0.88, 0, 'OBS(1,0)', fontsize=8)
    # plt.text(0.33, 0.4, 's1(x1,y1)', fontsize=8)
    # plt.text(0.07, 1, 's2(x2,y2)', fontsize=8)
    # plt.text(0.47, 0.95, 's3(x3,y3)', fontsize=8)

    # # 添加线段
    # plt.plot([dfmax[0, 0], dfmax[0, 0]], [dfmax[0, 1], dfmax[0, 1]], color='red', linewidth=2.5)
    # plt.plot([dfmax[0, 0], dfmax[1, 0]], [dfmax[0, 1], dfmax[1, 1]], color='blue', linewidth=2.5)
    # plt.plot([dfmax[0, 0], dfmax[2, 0]], [dfmax[0, 1], dfmax[2, 1]], color='black', linewidth=2.5)

    # plt.xlabel('CC')
    # plt.ylabel('NAE')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)

    # save_fig = os.path.join('D:/研究生/毕业论文/new毕业论文/预答辩/英那河/站点信息/CCHZ',  "results"+str(i)+".png")
    # plt.savefig(save_fig, bbox_inches="tight")
    # plt.close()