import yaml
from hydromodel.data.data_preprocess import split_train_test, cross_valid_data
from pathlib import Path
import numpy as np
from xaj.xajmodel import xaj_state as xaj_state

class configuration():
    
    def read_config(config_file):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        return config
    
    def process_inputs(forcing_file, json_file, npy_file, config):
        

        train_period = config['train_period']  
        test_period = config['test_period']
        period = config['period']
        json_file = Path(json_file)
        npy_file = Path(npy_file)
        split_train_test(json_file, npy_file, train_period, test_period)

        warmup_length = config['warmup_length']  
        cv_fold = config['cv_fold']  
        cross_valid_data(json_file, npy_file, period, warmup_length, cv_fold) 
        
    def extract_forcing(forcing_data):
        # p_and_e_df = forcing_data[["rainfall[mm]", "TURC [mm d-1]"]]
        p_and_e_df = forcing_data[["pre", "turc"]]
        p_and_e= np.expand_dims(p_and_e_df.values, axis=1)  
        return p_and_e_df, p_and_e

    def warmup(p_and_e_warmup,params_state, warmup_length):

        q_sim, es, *w0, w1, w2,s0, fr0, qi0, qg0 = xaj_state(              
                                        p_and_e_warmup,              
                                        params_state,           
                                        warmup_length= warmup_length,
                                        source_book="HF",
                                        source_type="sources",              
                                        return_state=True,            
                                        )
        return  q_sim,es,*w0, w1, w2, s0, fr0, qi0, qg0
        print(*w0, w1, w2, s0, fr0, qi0, qg0)
    def get_time_config(config):
        start_time_str = config['start_time_str'] 
        end_time_str = config['end_time_str']
        time_units = config['time_units']
        return start_time_str, end_time_str, time_units
    
