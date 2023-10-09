import os

import pandas as pd

import definitions
from hydromodel.utils.dmca_esr import step1_step2_tr_and_fluctuations_timeseries, step3_core_identification, \
    step4_end_rain_events, \
    step5_beginning_rain_events, step6_checks_on_rain_events, step7_end_flow_events, step8_beginning_flow_events, \
    step9_checks_on_flow_events, step10_checks_on_overlapping_events


def test_session_division_new():
    rain = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/division_rain.csv'))['rain']
    flow = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/division_flow.csv'))['flow']
    time = rain.index.to_numpy()
    rain = rain.to_numpy() / 24
    flow = flow.to_numpy() / 24
    rain_min = 0.02
    max_window = 100
    Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = step1_step2_tr_and_fluctuations_timeseries(rain, flow,
                                                                                                      rain_min,
                                                                                                      max_window)
    beginning_core, end_core = step3_core_identification(fluct_bivariate_Tr)
    end_rain = step4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min)
    beginning_rain = step5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min)
    beginning_rain_checked, end_rain_checked, beginning_core, end_core = step6_checks_on_rain_events(beginning_rain,
                                                                                                     end_rain, rain,
                                                                                                     rain_min,
                                                                                                     beginning_core,
                                                                                                     end_core)
    end_flow = step7_end_flow_events(end_rain_checked, beginning_core, end_core, rain, fluct_rain_Tr, fluct_flow_Tr, Tr)
    beginning_flow = step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, rain, beginning_core,
                                                 fluct_rain_Tr,
                                                 fluct_flow_Tr)
    beginning_flow_checked, end_flow_checked = step9_checks_on_flow_events(beginning_rain_checked, end_rain_checked,
                                                                           beginning_flow,
                                                                           end_flow, fluct_flow_Tr)
    BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW = step10_checks_on_overlapping_events(beginning_rain_checked,
                                                                                             end_rain_checked,
                                                                                             beginning_flow_checked,
                                                                                             end_flow_checked,
                                                                                             time)
    print(BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW)
    print(len(BEGINNING_RAIN), len(END_RAIN), len(BEGINNING_FLOW), len(END_FLOW))
