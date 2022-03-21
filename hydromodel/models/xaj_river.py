"""TODO: river routing process"""
import numpy as np


def network_route(runoffs, route_params):
    """
    河网汇流计算，新安江模型里一般采用线性水库或滞后演算法。这里使用滞后演算法

    Parameters
    ----------
    runoffs
        坡面汇流计算结果，数组
    route_params
        模型参数

    Returns
    -------
    np.array
        汇流计算结果——流量过程线
    """
    # 取整后面才能计算
    t = int(route_params["L"])
    cr = route_params["CR"]
    # 初始化
    qr = runoffs
    q = np.zeros(runoffs.size)
    if t <= 0:
        for i in range(runoffs.size):
            if i == 0:
                q[0] = (1 - cr) * qr[0]
            else:
                q[i] = cr * q[i - 1] + (1 - cr) * qr[i]
    else:
        for i in range(runoffs.size):
            if i == 0:
                q[0] = 0
            elif i < t:
                q[i] = cr * q[i - 1]
            else:
                q[i] = cr * q[i - 1] + (1 - cr) * qr[i - t]
    return q


def river_route(config, route_params, qf):
    """
    河道汇流计算，新安江模型一般采用马斯京根法

    Parameters
    ----------
    config
        计算设置条件
    route_params
        模型参数
    qf
        河网汇流计算结果，数组

    Returns
    -------
    np.array
        汇流计算结果——流量过程线
    """
    ke = route_params["KE"]
    xe = route_params["XE"]
    time_interval = config["time_interval/h"]

    q = np.zeros(qf.size)

    c0 = (0.5 * time_interval - ke * xe) / (0.5 * time_interval + ke - ke * xe)
    c1 = (0.5 * time_interval + ke * xe) / (0.5 * time_interval + ke - ke * xe)
    c2 = 1 - c0 - c1

    q[0] = qf[0]
    if c0 >= 0 and c2 >= 0:
        for i in range(1, q.size):
            q[i] = c0 * qf[i] + c1 * qf[i - 1] + c2 * q[i - 1]
    else:
        # 当马斯京根不适用时，暂未处理
        q = qf
    return q


# -------------------------------------------Unfinished auxiliary function---------------------------------------------
def uh_recognise(runoffs, flood_data):
    """
    时段单位线的识别，先以最小二乘法为主。

    Parameters
    ----------
    runoffs
        各场次洪水对应的各时段净雨，np.array组成的list
    flood_data
        多组出口断面流量过程线，np.array组成的list

    Returns
    -------
    np.array
        时段单位线，每个数据的单位为m^3/s
    """
    # 最小二乘法计算针对每场次洪水得到一条单位线，多场次洪水，按照书上的意思，可以取平均。如果曲线之间差异较大，需要进行分类，目前先求平均。
    uh_s = []
    uh_sum = np.array([])
    for i in range(len(runoffs)):
        ht = []
        # q表示实际径流
        q = flood_data[i]
        l = len(flood_data[i])
        m = len(runoffs[i])
        n = l - m + 1
        for j in range(n):
            # numpy默认为行向量
            h_row = np.zeros(l)
            for k in range(j, j + m):
                h_row[k] = runoffs[i][k - j]
            ht.append(h_row)
        h = np.transpose(ht)
        ht_h = np.dot(ht, h)
        ht_q = np.dot(ht, q)
        # 求得一条单位线
        uh_temp = np.linalg.solve(ht_h, ht_q)
        # 每场次洪水均有一条单位线
        uh_s.append(uh_temp)
        # 求和，因为单位线长度可能不一致，所以需要进行补0对齐
        if i == 0:
            uh_sum = uh_temp
        else:
            # 当维度不同的向量要对齐时，需在不足处补0
            length_zero = max(uh_sum.size, uh_temp.size) - min(
                uh_sum.size, uh_temp.size
            )
            zeros_need = np.zeros(length_zero)
            if uh_sum.size > uh_temp.size:
                arr_new = np.hstack([uh_temp, zeros_need])
                uh_sum = uh_sum + arr_new
            else:
                arr_new = np.hstack([uh_sum, zeros_need])
                uh_sum = uh_temp + arr_new
    # 广播运算
    uh = uh_sum / len(runoffs)
    return uh


def split_flow(floods_data, auto=None):
    """
    分割流量过程，有时候，会手动进行分割，所以可能不需要调用次函数

    Parameters
    ----------
    floods_data
        多组观测的出口断面流量过程线，np.array组成的list
    auto
        是程序自动计算退水曲线还是已经手动计算好了

    Returns
    -------
    list[np.array]
        去掉前期洪水尚未退完的部分水量和非本次降雨补给的流量过程线
    """
    # TODO: 求退水曲线
    # 首先取出各个洪水过程退水段
    floods = []
    for flood_data in floods_data:
        temp = flood_data.loc[:, "flood_quant"]
        floods.append(np.array(temp))
    # 然后取出退水段各点，组成向量

    # 最小二乘法计算

    # 利用计算好的退水曲线分割观测的流量过程线

    if auto is None:
        return floods
    return floods


def divide_source(floods, auto=None):
    """
    从观测的场次洪水径流中分割地表地下径流

    Parameters
    ----------
    floods
        多组观测的已经去掉前期洪水尚未退完的部分水量和非本次降雨补给的出口断面流量过程线，np.array组成的list
    auto
        是程序自动计算分割流量还是已经手动计算好了

    Returns
    -------
    list[np.array]
        rs_s,rg_s: 地表径流流量过程线和地下径流过程线
    """
    # TODO: 分割径流
    if auto is None:
        return floods
    return floods


def iuh_recognise(
    runoffs, flood_data, linear_reservoir=None, linear_canal=None, isochrone=None
):
    """
    瞬时单位线的识别，目前以计算Nash单位线为主

    Parameters
    ----------
    runoffs
        多组各时段净雨（地表），矩阵表示
    flood_data
        多组出口断面流量过程线，矩阵表示
    linear_reservoir

    linear_canal
        线性渠个数
    isochrone
        等流时线个数

    Returns
    -------
    tuple[float, float]
        n,k:nash单位线两参数
    """
    # TODO: 这个函数没想好怎么写，能很好地把串并联结构和三个基础模型搭配在一起，目前以n个线性水库串联为主

    return
