import numpy as np
import xarray as xr
from hydromodel.models.xaj1 import xaj1
from hydromodel.models.musk import Musk
from hydromodel.models.model_config import read_model_param_dict
import json
from hydromodel.datasets.data_preprocess import _get_pe_q_from_ts, cross_val_split_tsdata
import logging  # 去除debug信息
logging.basicConfig(level=logging.WARNING)


# attributes = xr.open_dataset('L:/XAJMUSK/NEW/input/attributes.nc')  # 读取流域属性数据
# with open('input/topo.txt', 'r') as f:
#     topo = f.readlines()  # 加载拓扑
# with open('L:/XAJMUSK/NEW/input/ModelwithsameParas.json', 'r', encoding='utf-8') as file:
#     modelwithsameParas = json.load(file)  # 加载率定参数
# with open('L:/XAJMUSK/NEW/input/params.json', 'r', encoding='utf-8') as file:
#     params_range = json.load(file)  # 加载所有参数

# para_seq = np.random.rand(sum(len(item['PARAMETER']) for item in modelwithsameParas))  # 长序列参数



def semi_xaj(p_and_e, attributes, modelwithsameParas, para_seq, params_range, topo, dt, **kwargs):
    model_name = kwargs.get("name", "xaj")
    source_type = kwargs.get("source_type", "sources")
    source_book = kwargs.get("source_book", "HF")
    qsim_collect = np.zeros((len(p_and_e),len(topo),1))
    print(para_seq.shape, '------------------------------------')
    
    # 把长序列参数分配给各个ParaID
    start_index = 0
    for item in modelwithsameParas:
        param_length = len(item['PARAMETER'])
        item['PARAMETER'] = para_seq[start_index:start_index + param_length].tolist()
        start_index += param_length
    
    for i in range(len(modelwithsameParas)):
        modelIdSet=modelwithsameParas[i]['MODELIDSET']
        for j in range(len(modelIdSet)):
            params_range[modelIdSet[j]-1]['PARAMETER'] = modelwithsameParas[i]['PARAMETER']
            params_range[modelIdSet[j]-1]['UP'] = modelwithsameParas[i]['UP']
            params_range[modelIdSet[j]-1]['DOWN'] = modelwithsameParas[i]['DOWN']
    
    lineN0=0
    for calid in topo:
        
        topovalue = calid.split()
        numbers = np.array([int(num) for num in topovalue])  # 每一行的拓扑值
        p_and_e1 = p_and_e[:,lineN0 :lineN0+1, :]
        lineN0=lineN0+1
        
        for i in range(len(numbers)):
            start, end = numbers[i], numbers[0]
            modelid = [model["MODELID"] for model in params_range if model["START"] == start and model["END"] == end]
            modelname = params_range[modelid[0]-1]['MODELNAME']

            if modelname == 'XAJ':
                print(f'Running XAJ')
                parameter = np.array(params_range[modelid[0]-1]['PARAMETER'])
                parameterup = np.array(params_range[modelid[0]-1]['UP'])
                parameterdown = np.array(params_range[modelid[0]-1]['DOWN'])
                parameter_xaj = (parameterup - parameterdown) * parameter + parameterdown
                parameter_xaj = parameter_xaj.reshape(-1, 1)
                print(attributes,'wwwwwwwwwwwwwwww')
                area = attributes.sel(id=str(numbers[0]))['area'].values

                qsim, _ = xaj1(
                    p_and_e1,
                    params=parameter_xaj,
                    warmup_length=0,
                    model_name=model_name,
                    source_type=source_type,
                    source_book=source_book, 
                    time_interval_hours=dt,
                )
                qsim = qsim.squeeze() * area / (3600*dt*1000/1000000)
                qsim_collect[:,numbers[0]-1,0] += qsim
                print(f'node:{start}-{end}\tmodel:{modelname}\tarea:{area}\tparameter:{parameter_xaj.squeeze()}')

            elif modelname=='MUSK':
                print(f'Running MUSK')
                parameter = np.array(params_range[modelid[0]-1]['PARAMETER'])
                inflows = qsim_collect[:,start-1,0]
                outflows = Musk(inflows, parameter[0], parameter[1], dt=dt)
                qsim_collect[:,end-1,0] += outflows
    return qsim_collect

