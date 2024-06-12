import numpy as np
import pandas as pd

def get_station_list(zip_file_list, df_detector_id, df_station_metadata):
    station_list = df_station_metadata['ID'].unique()
    for zip_file_name in zip_file_list:
        unzipped_file_name = zip_file_name[:-3]
        date = zip_file_name[22:32]
        df_combined = pd.read_csv("C:/Users/p309996/OneDrive - California Department of Transportation/Desktop/XInyue Project/data/sorted data tensor/"+unzipped_file_name[:-4]+'.csv')
        df_combined.set_index(df_combined.columns[0], inplace=True)

        filtered_list = [i for i in station_list if i in df_combined.index]
        station_list = filtered_list
    filtered_list = [i for i in station_list if i in df_detector_id.STATION_ID]
    station_list = filtered_list
    print("After filtering, the station number = ", len(station_list))
    return station_list

def get_detector_lane_dict(df_detector_id, detector_list):
    df_temp = df_detector_id.groupby(["DETECTOR_ID"]).last()
    df_temp = df_temp[df_temp.index.isin(detector_list)]
    return df_temp["LANE"].to_dict()

def get_detector_list(df_detector_id, station_list):
    df_temp = df_detector_id[["STATION_ID", "DETECTOR_ID"]].drop_duplicates()
    df_temp = df_temp[df_temp.STATION_ID.isin(station_list)]
    detector_list = []
    for station_id in station_list:
        detector_list.extend(list(df_temp[df_temp["STATION_ID"]==station_id]["DETECTOR_ID"]))
    station_to_detector_dict = df_temp.groupby('STATION_ID')['DETECTOR_ID'].apply(list).to_dict()

    # remove detectors that have changed lane id in the dataset 
    # it cause difficulty to retrieve the columns in original 5min datatable 
    df_temp = df_detector_id[df_detector_id["STATION_ID"].isin(station_list)]
    df_temp = df_temp[["DETECTOR_ID","STATION_ID","LANE"]].drop_duplicates()
    detector_list_changed_lane = df_temp.groupby('DETECTOR_ID').filter(lambda x: x['LANE'].nunique() > 1)['DETECTOR_ID'].unique()
    for detector in detector_list_changed_lane:
        detector_list.remove(detector)
    
    df_temp = df_detector_id[df_detector_id.DETECTOR_ID.isin(detector_list)]
    df_temp = df_temp[["DETECTOR_ID", "STATION_ID"]].drop_duplicates()
    station_to_detector_dict = df_temp.groupby('STATION_ID')['DETECTOR_ID'].apply(list).to_dict()
    station_list = list(station_to_detector_dict.keys())
    detector_list.sort()

    return station_list, detector_list, station_to_detector_dict
    
def get_station_dist_list(station_list, df_station_metadata):
    # station_list or I80
    station_dist_list = []
    for station_id in station_list:
        station_dist_list.append(float(df_station_metadata[df_station_metadata.ID == station_id].Abs_PM.iloc[0]))
    return station_dist_list



def get_detector_to_station_dict(station_list, station_to_detector_dict):
    # station_list or I80
    detector_to_station_dict = {}
    for station_id in station_list:
        for detector_id in station_to_detector_dict[station_id]:
            detector_to_station_dict[detector_id] = station_id
    return detector_to_station_dict


def get_station_list_by_highway(station_list, df_station_metadata):
    station_list_by_highway = []
    highway_list = []
    station_dist_list_by_highway = []
    # Get distinct combinations of values in 'Fwy' and 'Dir'
    distinct_combinations = df_station_metadata[['Fwy', 'Dir']].drop_duplicates()
    for combination in distinct_combinations.values:
        # Filter the original DataFrame based on the first combination
        df_temp = df_station_metadata[(df_station_metadata['Fwy'] == combination[0]) & 
                                    (df_station_metadata['Dir'] == combination[1])]
        filtered_station_list = [station_id for station_id in station_list if station_id in df_temp['ID'].tolist()]
        if len(filtered_station_list)>0:
            highway_list.append(list(combination))
            station_list_by_highway.append(filtered_station_list)
            station_dist_list_by_highway.append([float(df_station_metadata[df_station_metadata.ID == station_id].Abs_PM.iloc[0]) for station_id in filtered_station_list])
    return station_list_by_highway, highway_list, station_dist_list_by_highway

def get_station_neighbors(station_list_by_highway, station_dist_list_by_highway, max_dist):
    '''
    input: station_list, ex.[s1, s2, s3]
           station_dist_list_by_highway, ex.[0.5, 1.5, 4]
    for s2, the upstream is s1, downstream is s3: s2: [[s1, s4], [1, 2.5]]
    '''
    neighbor_stations_dict = {}
    for station_list, station_dist_list in zip(station_list_by_highway, station_dist_list_by_highway):
        # Zip lists and sort by distance
        sorted_pairs = sorted(zip(station_list, station_dist_list), key=lambda x: x[1])
        station_order_list, station_dist_list_sorted = zip(*sorted_pairs)
    
        station_order_list = list(station_order_list)
        station_dist_list_sorted = list(station_dist_list_sorted)
        # add nan at both sides
        station_order_list = [np.nan]+station_order_list+[np.nan]
        dist_diff = list(np.array(station_dist_list_sorted[1:]) - np.array(station_dist_list_sorted[:-1]))
        dist_diff = [np.nan]+dist_diff+[np.nan]
        for i, station_id in enumerate(station_order_list):
            if not np.isnan(station_id):
                neighbor_stations = [station_order_list[i-1], station_order_list[i+1]]
                dist_to_neighbors = [dist_diff[i-1], dist_diff[i]]
                if dist_diff[i-1] > 1:
                    neighbor_stations[0] = np.nan
                if dist_diff[i] > 1:
                    neighbor_stations[1] = np.nan
                    
                neighbor_stations_dict[station_id] = [neighbor_stations, dist_to_neighbors]
    return neighbor_stations_dict



def get_detector_neighbors(detector_list, station_to_detector_dict, detector_to_station_dict, neighbor_stations_dict, detector_lane_dict):
    neighbor_detectors_dict = {}
    for detector_id in detector_list:
        current_station_id = detector_to_station_dict[detector_id]
        upstream_station_id, downstream_station_id = neighbor_stations_dict[current_station_id][0]
        dist_to_upstream, dist_to_downstream = neighbor_stations_dict[current_station_id][1]
        # the upstream and downstream should be same lane if possible 
        current_detector_lane_num = detector_lane_dict[detector_id]
        
        # find upstream detector with the closest lane number
        if np.isnan(upstream_station_id):
            upstream_detector_id = np.nan
        else:
            lane_num_diff = 100 # arbitrary large number
            for id in station_to_detector_dict[upstream_station_id]:
                if abs(detector_lane_dict[id]-current_detector_lane_num)<=lane_num_diff:
                    lane_num_diff = abs(detector_lane_dict[id]-current_detector_lane_num)
                    upstream_detector_id = id
        
        # find downstream detector with the closest lane number
        if np.isnan(downstream_station_id):
            downstream_detector_id = np.nan
        else:    
            lane_num_diff = 100 # arbitrary large number
            for id in station_to_detector_dict[downstream_station_id]:
                if abs(detector_lane_dict[id]-current_detector_lane_num)<=lane_num_diff:
                    lane_num_diff = abs(detector_lane_dict[id]-current_detector_lane_num)
                    downstream_detector_id = id
        
        neighbor_detectors_dict[detector_id] = [upstream_detector_id, downstream_detector_id]
    return neighbor_detectors_dict
