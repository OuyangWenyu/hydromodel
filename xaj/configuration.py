import yaml
import numpy as np
from xaj.xajmodel import xaj_state as xaj_state


def read_config(config_file):
    with open(config_file, encoding='utf8') as f:
        config = yaml.safe_load(f)
    return config


def extract_forcing(forcing_data):
    # p_and_e_df = forcing_data[["rainfall[mm]", "TURC [mm d-1]"]]
    p_and_e_df = forcing_data[["prcp(mm/day)", "petfao56(mm/day)"]]
    p_and_e = np.expand_dims(p_and_e_df.values, axis=1)
    return p_and_e_df, p_and_e


def warmup(p_and_e_warmup, params_state, warmup_length, model_info):
    q_sim, es, *w0, w1, w2, s0, fr0, qi0, qg0 = xaj_state(
        p_and_e_warmup,
        params_state,
        warmup_length=warmup_length,
        model_info=model_info,
        return_state=True,
    )
    return q_sim, es, *w0, w1, w2, s0, fr0, qi0, qg0
