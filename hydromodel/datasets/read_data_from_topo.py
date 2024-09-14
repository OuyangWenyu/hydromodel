import geopandas as gpd
import sqlalchemy as sqa
import pandas as pd

from hydromodel.datasets import CODE_NAME, STTYPE_NAME


def read_data_from_topo(node_df: gpd.GeoDataFrame, sta_indexes):
    stcds = node_df.loc[sta_indexes, CODE_NAME].values
    sql_engine = sqa.create_engine('postgresql+psycopg2://student:student@10.55.0.102:5432/water')
    number_dict = {}
    for stcd in stcds:
        ind = node_df[node_df[CODE_NAME] == stcd].index[0]
        stcd_ = node_df[STTYPE_NAME][node_df[CODE_NAME] == stcd].values[0]
        if (stcd_ == 'ZZ') | (stcd_ == 'ZQ'):
            sql_command = f"SELECT * FROM ht_river_r WHERE stcd = '{stcd}'"
        elif stcd_ == 'RR':
            sql_command = f"SELECT * FROM ht_rsvr_r WHERE stcd = '{stcd}'"
        else:
            sql_command = f"SELECT * FROM ht_pptn_r WHERE stcd = '{stcd}'"
        df = pd.read_sql(sql_command, sql_engine)
        number_dict[ind] = df
    return number_dict



