# Data processing
import numpy as np
import pandas as pd
import datetime 
import time
from datetime import datetime, timedelta

def initialize_zip_file_list(start_date, end_date):
    global zip_file_list  # Access the global zip_file_list variable
    zip_file_list = create_zip_file_list(start_date, end_date)
    return(zip_file_list)

def create_zip_file_list(start_date, end_date):
    zip_file_list = list()
    current_date = start_date
    while current_date <= end_date: 
        zip_file_list.append("d03_text_station_5min_"+current_date.strftime("%Y_%m_%d")+".txt.gz")
        current_date += timedelta(days=1)
    return zip_file_list


def find_station_list(df_station_metadata, df_detector_id):
    ''''
    NEED TO FIX
    '''
    station_list = df_station_metadata['ID'].unique()

    print("Before filtering, the sensor number = ", len(station_list))
    for zip_file_name in zip_file_list:
        unzipped_file_name = zip_file_name[:-3]
        date = zip_file_name[22:32]
        df_combined = pd.read_csv("C:/Users/p309996/OneDrive - California Department of Transportation/Desktop/XInyue Project/data/sorted data tensor/"+unzipped_file_name[:-4]+'.csv')
        df_combined.set_index(df_combined.columns[0], inplace=True)
        filtered_list = [i for i in station_list if i in df_combined.index]
        station_list = filtered_list
        filtered_list = [i for i in station_list if i in df_detector_id.STATION_ID]
        station_list = filtered_list
    print("After filtering, the sensor number = ", len(station_list))
    return station_list


def data_convert_ordered(from_folder, to_folder, station_list, station_to_detector_dict, detector_lane_dict):
    '''
    This function takes the downloaded zipped data (d03_text_station_5min) files and unzip
    to a target folder.
    start_date: datetime obj, ex. datetime.datetime(2023, 7, 1)
    
    end_date: datetime obj ex datetime.datetime(2024, 3, 31)
    
    from_folder: str, ex. "C:/Users/p309996/OneDrive - California Department of Transportation\Desktop/XInyue Project/data/unzipped data/"
    
    to_folder: str, ex. "C:/Users/p309996/OneDrive - California Department of Transportation/Desktop/XInyue Project/data/sorted data tensor/"
    
    station_list: a list of station_IDs that always contains data within the timeframe 

    df_detector_id: pd.DataFrame
    '''

    for zip_file_name in zip_file_list:
        unzipped_file_name = zip_file_name[:-3]
        df_temp = pd.read_csv(from_folder+unzipped_file_name, header=None)
        df_temp = df_temp[(df_temp[5] == "ML")]

        df_detector_combined = pd.DataFrame()

        for station_id in station_list:
            df_temp_station = df_temp[df_temp[1]==station_id]
            for detector_id in station_to_detector_dict[station_id]:
                lane_number = detector_lane_dict[detector_id]
                df_temp_detector = df_temp_station.loc[:,[0,5*lane_number+9, 5*lane_number+11]]  #9 for occupancy 
                df_temp_detector["DETECTOR_ID"] = detector_id
                new_column_names = ['time', 'occ', 'observed', 'DETECTOR_ID']  # List of new column names
                df_temp_detector.columns = new_column_names
                df_detector_combined =  pd.concat([df_detector_combined, df_temp_detector], axis=0)


        df_combined = df_detector_combined.pivot(index="DETECTOR_ID", columns="time", values="occ")
        df_index = df_detector_combined.pivot(index="DETECTOR_ID", columns="time", values="observed")

        ############################ fixing non existing time stamps in dataset #######################
        date_string = df_temp[0][0][0:11]
        start_timestamp = datetime.strptime(date_string+'00:00:00', '%m/%d/%Y %H:%M:%S')
        end_timestamp = datetime.strptime(date_string+'23:55:00', '%m/%d/%Y %H:%M:%S')

        # Define the interval (5 minutes)
        interval = timedelta(minutes=5)

        # Generate the list of timestamps
        timestamps = []
        current_timestamp = start_timestamp
        while current_timestamp <= end_timestamp:
            timestamps.append(current_timestamp.strftime('%m/%d/%Y %H:%M:%S'))
            current_timestamp += interval
        ########################################3
        for col in timestamps:
            if col not in df_combined.columns:
                df_combined[col] = 0
                df_index[col] = 0
                
        df_combined.to_csv(to_folder+unzipped_file_name[:-4]+'.csv')
        df_index.to_csv(to_folder+unzipped_file_name[:-4]+'_index.csv')

        print("Finished ---", unzipped_file_name)