import hydrotopo.ig_path as htip
import numpy as np


# 特化，只针对上游
def find_edge_nodes(gpd_nodes_df, gpd_network_df, station_indexes, cutoff: int = 2):
    geom_array, new_geom_array, index_geom_array = htip.line_min_dist(gpd_nodes_df, gpd_network_df)
    graph = htip.build_graph(geom_array)
    station_dict = {}
    # 当前站点所对应线索引
    for station_index in station_indexes:
        cur_index = np.argwhere(new_geom_array == index_geom_array[station_index])[0][0]
        true_index = len(geom_array) - len(new_geom_array) + cur_index
        paths = graph.get_all_shortest_paths(v=true_index, mode='in')
        sta_lists = []
        for path in paths:
            sta_list = []
            for line in path:
                if line >= len(geom_array) - len(new_geom_array):
                    new_line_index = line - len(geom_array) + len(new_geom_array)
                    sta_index = np.argwhere(index_geom_array == new_geom_array[new_line_index])
                    if len(sta_index) > 0:
                        sta_list.append(sta_index[0][0])
            sta_list.reverse()
            sta_lists.append(sta_list[-cutoff:])
        paths = np.unique(np.array(sta_lists, dtype=object))
        station_dict[station_index] = paths
    return station_dict

def gen_topo_text(gpd_nodes_df, gpd_network_df, station_indexes):
    station_dict = find_edge_nodes(gpd_nodes_df, gpd_network_df, station_indexes)
    riv_1lvl_list = []
    higher_list = []
    for val in station_dict.values():
        if len(val) == 1:
            riv_1lvl_list.extend(val)
        else:
            topo_path = np.unique(np.concatenate(val))
            higher_list.append(topo_path)
    with open('topo.txt', 'w') as f:
        [f.write(str(i)+' ') for i in riv_1lvl_list]
        f.write('\n')
        for hlist in higher_list:
            [f.write(str(i)+' ') for i in hlist]
            f.write('\n')

