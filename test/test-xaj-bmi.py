import logging
logging.basicConfig(level=logging.INFO)
from xaj_bmi import xajBmi
import pandas as pd
# from test.test_xaj import test_xaj
# from configuration import configuration
# import numpy as np
model = xajBmi()
print(model.get_component_name())


model.initialize("xaj/runxaj.yaml")
print("Start time:", model.get_start_time())
print("End time:", model.get_end_time())
print("Current time:", model.get_current_time())
print("Time step:", model.get_time_step())
print("Time units:", model.get_time_units())
print(model.get_input_var_names())
print(model.get_output_var_names())

discharge = []
ET = []
time = []                                          
while model.get_current_time() <= model.get_end_time():
    time.append(model.get_current_time())
    model.update()

discharge=model.get_value("discharge")
ET=model.get_value("ET")

results = pd.DataFrame({
                'discharge': discharge.flatten(),
                'ET': ET.flatten(),  
            })
results.to_csv('/home/wangjingyi/code/hydro-model-xaj/scripts/xaj.csv')
model.finalize()
# params=np.tile([0.5], (1, 15))
# config = configuration.read_config("scripts/runxaj.yaml")
# forcing_data = pd.read_csv(config['forcing_file'])
# p_and_e_df, p_and_e = configuration.extract_forcing(forcing_data)
# test_xaj(p_and_e=p_and_e,params=params,warmup_length=360)


