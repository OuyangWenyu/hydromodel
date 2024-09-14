import geopandas as gpd
import hydromodel.datasets.separate_stations as hdss

def test_point_stations_polygons():
    stations = gpd.read_file(r'C:\Users\Administrator\IdeaProjects\hydromodel-master\input\station_shps\biliu_21401550.shp')
    basin = gpd.read_file(r'C:\Users\Administrator\IdeaProjects\hydromodel-master\input\station_shps\basin_CHN_songliao_21401550.shp')
    stations_npp = stations[stations['类型'].str.contains('|'.join(['RR', 'ZQ']))]
    stations_pp = stations[stations['类型'] == 'PP']

    polygons = hdss.calculate_voronoi_polygons(stations_npp, basin)
    nearest_gdf = gpd.sjoin_nearest(stations_pp, polygons, how="left")
    # polygons.to_file(r'C:\Users\Administrator\IdeaProjects\hydromodel-master\input\station_shps\polygons.shp')
    return nearest_gdf
