import os
from typing import Union
import numpy as np
import spotpy
import pandas as pd
from spotpy.parameter import Uniform, ParameterSet
from hydromodel.models.model_config import read_model_param_dict
from hydromodel.models.model_dict import LOSS_DICT, MODEL_DICT


class SpotSetup(object):
    def __init__(
        self,calibrate_id, p_and_e, qobs, attributes, modelwithsameParas, params_range, topo, dt, warmup_length=365, model=None, loss=None
    ):
        self.calibrate_id = calibrate_id #确定校核断面
        self.attributes = attributes
        self.modelwithsameParas = modelwithsameParas
        self.params_range = params_range
        self.topo = topo
        self.dt = dt
        self.model = model
        self.para_seq = []
        para_number = sum(len(item['PARAMETER']) for item in self.modelwithsameParas)
        self.para_seq.extend(Uniform(f'param_{i}', low=0.0, high=1.0) for i in range(para_number))
        self.loss = loss
        self.p_and_e = p_and_e
        self.true_obs = qobs[warmup_length:, :, :]
        self.warmup_length = warmup_length

    def parameters(self):
        # print(f'参数：{spotpy.parameter.generate(self.para_seq)}')
        return spotpy.parameter.generate(self.para_seq)
    

    def simulation(self, x: ParameterSet) -> Union[list, np.array]:
        # parameter, 2-dim variable: [basin=1, parameter]
        param_seq = np.array(x)
        calibrate_id=self.calibrate_id
        sim = MODEL_DICT[self.model["name"]](
            self.p_and_e,
            self.attributes, self.modelwithsameParas, param_seq, self.params_range, self.topo, self.dt,
            warmup_length=self.warmup_length,
            **self.model,
        )
        area = self.attributes.sel(id=str(calibrate_id))['area'].values

        return sim[:,calibrate_id,:] / area * (3600*self.dt*1000/1000000)

    def evaluation(self) -> Union[list, np.array]:
        return self.true_obs

    def objectivefunction(
        self,
        simulation: Union[list, np.array],
        evaluation: Union[list, np.array],
        params=None,  # this cannot be removed
    ) -> float:
        if self.loss["type"] == "time_series":
            return LOSS_DICT[self.loss["obj_func"]](evaluation, simulation)
        # for events
        time = self.loss["events"]
        if time is None:
            raise ValueError(
                "time should not be None since you choose events, otherwise choose time_series"
            )
        # TODO: not finished for events
        calibrate_starttime = pd.to_datetime("2012-06-10 0:00:00")
        calibrate_endtime = pd.to_datetime("2019-12-31 23:00:00")
        total = 0
        count = 0
        for i in range(len(time)):
            if time.iloc[i, 0] < calibrate_endtime:
                start_num = (
                    time.iloc[i, 0] - calibrate_starttime - pd.Timedelta(hours=365)
                ) / pd.Timedelta(hours=1)
                end_num = (
                    time.iloc[i, 1] - calibrate_starttime - pd.Timedelta(hours=365)
                ) / pd.Timedelta(hours=1)
                start_num = int(start_num)
                end_num = int(end_num)
                like_ = LOSS_DICT[self.loss["obj_func"]](
                    evaluation[start_num:end_num,], simulation[start_num:end_num,]
                )
                count += 1

                total += like_
        return total / count


# semi_xaj(p_and_e, attributes, modelwithsameParas, para_seq, params_range, topo, dt)
def calibrate_semi_xaj_sceua(
    calibrate_id,
    basins,
    p_and_e,
    qobs,
    attributes, modelwithsameParas, params_range, topo, dt,
    dbname,
    warmup_length=365,
    model=None,
    algorithm=None,
    loss=None,
):
    random_seed = algorithm["random_seed"]
    rep = algorithm["rep"]
    ngs = algorithm["ngs"]
    kstop = algorithm["kstop"]
    peps = algorithm["peps"]
    pcento = algorithm["pcento"]
    np.random.seed(random_seed)
    for i in range(calibrate_id,calibrate_id+1):
        spot_setup = SpotSetup(
            calibrate_id,
            p_and_e[:, :, :],
            qobs[:, i : i + 1, :],
            attributes, modelwithsameParas, params_range, topo, dt,
            warmup_length=warmup_length,
            model=model,
            loss=loss,
        )
        if not os.path.exists(dbname):
            os.makedirs(dbname)
        db_basin = os.path.join(dbname, basins[i])
        sampler = spotpy.algorithms.sceua(
            spot_setup,
            dbname=db_basin,
            dbformat="csv",
            random_state=random_seed,
        )
        sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)
        print("Calibrate Finished!")
    return sampler
