import numpy as np


def movmean(X, n):
    ones = np.ones(X.shape)
    kernel = np.ones(n)
    return np.convolve(X, kernel, mode='same') / np.convolve(ones, kernel, mode='same')


def step1_step2_tr_and_fluctuations_timeseries(rain, flow, rain_min, max_window):
    """
    :param rain: 降雨量向量，单位mm/h，需注意与mm/day之间的单位转化
    :param flow: 径流量向量，单位m³/h，需注意与m³/day之间的单位转化
    :param rain_min: 最小降雨量阈值
    :param max_window: 场次划分最大窗口，决定场次长度
    """
    rain = rain.T
    flow = flow.T
    rain_int = np.nancumsum(rain)
    flow_int = np.nancumsum(flow)
    T = rain.size
    rain_mean = np.empty(((max_window - 1) // 2, T))
    flow_mean = np.empty(((max_window - 1) // 2, T))
    fluct_rain = np.empty(((max_window - 1) // 2, T))
    fluct_flow = np.empty(((max_window - 1) // 2, T))
    F_rain = np.empty((max_window - 1) // 2)
    F_flow = np.empty((max_window - 1) // 2)
    F_rain_flow = np.empty((max_window - 1) // 2)
    rho = np.empty((max_window - 1) // 2)
    for window in np.arange(3, max_window + 1, 2):
        int_index = int((window - 1) / 2 - 1)
        start_slice = int(window - 0.5 * (window - 1))
        dst_slice = int(T - 0.5 * (window - 1))
        # 新建一个循环体长度*数据长度的大数组
        rain_mean[int_index] = movmean(rain_int, window)
        flow_mean[int_index] = movmean(flow_int, window)
        fluct_rain[int_index] = rain_int - rain_mean[int_index, :]
        F_rain[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_rain[int_index, start_slice:dst_slice]) ** 2)
        fluct_flow[int_index, np.newaxis] = flow_int - flow_mean[int_index, :]
        F_flow[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_flow[int_index, start_slice:dst_slice]) ** 2)
        F_rain_flow[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_rain[int_index, start_slice:dst_slice]) * (
                fluct_flow[int_index, start_slice:dst_slice]))
        rho[int_index] = F_rain_flow[int_index] / (
                np.sqrt(F_rain[int_index]) * np.sqrt(F_flow[int_index]))
    pos_min = np.argmin(rho)
    Tr = pos_min + 1
    tol_fluct_rain = (rain_min / (2 * Tr + 1)) * Tr
    tol_fluct_flow = flow_int[-1] / 1e15
    fluct_rain[pos_min, np.fabs(fluct_rain[pos_min, :]) < tol_fluct_rain] = 0
    fluct_flow[pos_min, np.fabs(fluct_flow[pos_min, :]) < tol_fluct_flow] = 0
    fluct_rain_Tr = fluct_rain[pos_min, :]
    fluct_flow_Tr = fluct_flow[pos_min, :]
    fluct_bivariate_Tr = fluct_rain_Tr * fluct_flow_Tr
    fluct_bivariate_Tr[np.fabs(fluct_bivariate_Tr) < np.finfo(np.float64).eps] = 0  # 便于比较
    return Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr


def step3_core_identification(fluct_bivariate_Tr):
    d = np.diff(fluct_bivariate_Tr, prepend=[0], append=[0])  # 计算相邻数值差分，为0代表两端点处于0区间
    d[np.fabs(d) < np.finfo(np.float64).eps] = 0  # 确保计算正确
    d = np.logical_not(d)  # 求0-1数组，为真代表为0区间
    d0 = np.logical_not(np.convolve(d, [1, 1], 'valid'))  # 对相邻元素做OR，代表原数组数值是否处于某一0区间，再取反表示取有效值
    valid = np.logical_or(fluct_bivariate_Tr, d0)  # 有效core
    d_ = np.diff(valid, prepend=[0], append=[0])  # 求差分方便取上下边沿
    beginning_core = np.argwhere(d_ == 1)  # 上边沿为begin
    end_core = np.argwhere(d_ == -1) - 1  # 下边沿为end
    return beginning_core, end_core


def step4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min):
    end_rain = end_core.copy()
    rain = rain.T
    for g in range(end_core.size):
        if end_core[g] + 2 < fluct_rain_Tr.size and \
                (np.fabs(fluct_rain_Tr[end_core[g] + 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[end_core[g] + 2]) < np.finfo(np.float64).eps):
            # case 1&2
            if np.fabs(rain[end_core[g]]) < np.finfo(np.float64).eps:
                # case 1
                while end_rain[g] > beginning_core[g] and np.fabs(rain[end_rain[g]]) < np.finfo(np.float64).eps:
                    end_rain[g] = end_rain[g] - 1
            else:
                # case 2
                bound = beginning_core[g + 1] if g + 1 < beginning_core.size else rain.size
                while end_rain[g] < bound and rain[end_rain[g]] > rain_min:
                    end_rain[g] = end_rain[g] + 1
                end_rain[g] = end_rain[g] - 1  # 回到最后一个
        else:
            # case 3
            # 若在降水，先跳过
            while end_rain[g] >= beginning_core[g] and rain[end_rain[g]] > rain_min:
                end_rain[g] = end_rain[g] - 1
            while end_rain[g] >= beginning_core[g] and rain[end_rain[g]] < rain_min:
                end_rain[g] = end_rain[g] - 1
    return end_rain


def step5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min):
    beginning_rain = beginning_core.copy()
    rain = rain.T
    for g in range(beginning_core.size):
        if beginning_core[g] - 2 >= 0 \
                and (np.fabs(fluct_rain_Tr[beginning_core[g] - 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[beginning_core[g] - 2]) < np.finfo(
            np.float64).eps) \
                and np.fabs(rain[beginning_core[g]]) < np.finfo(np.float64).eps:
            # case 1
            while beginning_rain[g] < end_rain[g] and np.fabs(rain[beginning_rain[g]]) < np.finfo(np.float64).eps:
                beginning_rain[g] = beginning_rain[g] + 1
        else:
            # case 2&3
            bound = end_rain[g - 1] if g - 1 >= 0 else -1
            while beginning_rain[g] > bound and rain[beginning_rain[g]] > rain_min:
                beginning_rain[g] = beginning_rain[g] - 1
            beginning_rain[g] = beginning_rain[g] + 1  # 回到第一个
    return beginning_rain


def step6_checks_on_rain_events(beginning_rain, end_rain, rain, rain_min, beginning_core, end_core):
    rain = rain.T
    beginning_rain = beginning_rain.copy()
    end_rain = end_rain.copy()
    if beginning_rain[0] == 0:  # 掐头
        beginning_rain = beginning_rain[1:]
        end_rain = end_rain[1:]
        beginning_core = beginning_core[1:]
        end_core = end_core[1:]
    if end_rain[-1] == rain.size - 1:  # 去尾
        beginning_rain = beginning_rain[:-2]
        end_rain = end_rain[:-2]
        beginning_core = beginning_core[:-2]
        end_core = end_core[:-2]
    error_time_reversed = beginning_rain > end_rain
    error_wrong_delimiter = np.logical_or(rain[beginning_rain - 1] > rain_min, rain[end_rain + 1] > rain_min)
    beginning_rain[error_time_reversed] = -2
    beginning_rain[error_wrong_delimiter] = -2
    end_rain[error_time_reversed] = -2
    end_rain[error_wrong_delimiter] = -2
    beginning_core[error_time_reversed] = -2
    beginning_core[error_wrong_delimiter] = -2
    end_core[error_time_reversed] = -2
    end_core[error_wrong_delimiter] = -2
    beginning_rain = beginning_rain[beginning_rain != -2]
    end_rain = end_rain[end_rain != -2]
    beginning_core = beginning_core[beginning_core != -2]
    end_core = end_core[end_core != -2]
    return beginning_rain, end_rain, beginning_core, end_core


def step7_end_flow_events(end_rain_checked, beginning_core, end_core, rain, fluct_rain_Tr, fluct_flow_Tr, Tr):
    end_flow = np.empty(end_core.size, dtype=int)
    for g in range(end_rain_checked.size):
        if end_core[g] + 2 < fluct_rain_Tr.size and \
                (np.fabs(fluct_rain_Tr[end_core[g] + 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[end_core[g] + 2]) < np.finfo(np.float64).eps):
            # case 1
            end_flow[g] = end_rain_checked[g]
            bound = beginning_core[g + 1] + Tr if g + 1 < beginning_core.size else rain.size
            bound = min(bound, rain.size)  # 防溢出
            # 若flow为负，先跳过
            while end_flow[g] < bound and fluct_flow_Tr[end_flow[g]] <= 0:
                end_flow[g] = end_flow[g] + 1
            while end_flow[g] < bound and fluct_flow_Tr[end_flow[g]] > 0:
                end_flow[g] = end_flow[g] + 1
            end_flow[g] = end_flow[g] - 1  # 回到最后一个
        else:
            # case 2
            end_flow[g] = end_core[g]
            while end_flow[g] >= beginning_core[g] and fluct_flow_Tr[end_flow[g]] <= 0:
                end_flow[g] = end_flow[g] - 1
    return end_flow


def step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, rain, beginning_core, fluct_rain_Tr, fluct_flow_Tr):
    beginning_flow = np.empty(beginning_rain_checked.size, dtype=int)
    for g in range(beginning_rain_checked.size):
        if beginning_core[g] - 2 >= 0 \
                and (np.fabs(fluct_rain_Tr[beginning_core[g] - 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[beginning_core[g] - 2]) < np.finfo(
            np.float64).eps):
            beginning_flow[g] = beginning_rain_checked[g]  # case 1
        else:
            beginning_flow[g] = beginning_core[g]  # case 2
        while beginning_flow[g] < end_rain_checked[g] and fluct_flow_Tr[beginning_flow[g]] >= 0:
            beginning_flow[g] = beginning_flow[g] + 1
    return beginning_flow


def step9_checks_on_flow_events(beginning_rain_checked, end_rain_checked, beginning_flow, end_flow, fluct_flow_Tr):
    error_time_reversed = beginning_flow > end_flow
    error_wrong_fluct = np.logical_or(np.logical_or(fluct_flow_Tr[beginning_flow] > 0, fluct_flow_Tr[end_flow] < 0), np.logical_or(beginning_flow <
                                                                                                                                   beginning_rain_checked, end_flow < end_rain_checked))
    beginning_flow[error_time_reversed] = -3
    beginning_flow[error_wrong_fluct] = -3
    end_flow[error_time_reversed] = -3
    end_flow[error_wrong_fluct] = -3
    beginning_flow = beginning_flow[beginning_flow != -3]
    end_flow = end_flow[end_flow != -3]
    return beginning_flow, end_flow


def step10_checks_on_overlapping_events(beginning_rain_ungrouped, end_rain_ungrouped, beginning_flow_ungrouped, end_flow_ungrouped, time):
    # rain
    order1 = np.reshape(np.hstack((np.reshape(beginning_rain_ungrouped, (-1, 1)),
                                   np.reshape(end_rain_ungrouped, (-1, 1)))), (1, -1))
    reversed1 = np.diff(order1) <= 0
    order1[np.hstack((reversed1, [[False]]))] = -2
    order1[np.hstack(([[False]], reversed1))] = -2
    order1 = order1[order1 != -2]
    # flow
    order2 = np.reshape(np.hstack((np.reshape(beginning_flow_ungrouped, (-1, 1)),
                                   np.reshape(end_flow_ungrouped, (-1, 1)))), (1, -1))
    reversed2 = np.diff(order2) <= 0
    order2[np.hstack((reversed2, [[False]]))] = -3
    order2[np.hstack(([[False]], reversed2))] = -3
    order2 = order2[order2 != -3]
    # group
    rain_grouped = np.reshape(order1, (-1, 2)).T
    beginning_rain_grouped = rain_grouped[0]
    end_rain_grouped = rain_grouped[1]
    flow_grouped = np.reshape(order2, (-1, 2)).T
    beginning_flow_grouped = flow_grouped[0]
    end_flow_grouped = flow_grouped[1]
    return time[beginning_rain_grouped], time[end_rain_grouped], time[beginning_flow_grouped], time[end_flow_grouped]