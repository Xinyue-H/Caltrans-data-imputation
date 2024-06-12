import numpy as np
import pandas as pd

def generate_data_tensor(zip_file_list, detector_list):
    dfs_values = []  # Placeholder for DataFrames
    dfs_indices = []
    data_folder = "C:/Users/p309996/OneDrive - California Department of Transportation\Desktop/XInyue Project/ML project/Modules/data/processed_data/"

    for zip_file_name in zip_file_list:
        unzipped_file_name = zip_file_name[:-3]
        df_detector_combined = pd.read_csv(data_folder+unzipped_file_name[:-4]+'.csv', index_col=0)
        df_index = pd.read_csv(data_folder+unzipped_file_name[:-4]+'_index.csv', index_col=0)
        
        dfs_values.append(df_detector_combined.loc[detector_list].values)
        dfs_indices.append(df_index.loc[detector_list].values)

    dfs_values = np.array(dfs_values).astype(float)
    dfs_indices = np.array(dfs_indices)
    dfs_values[np.isnan(dfs_values)] = 0
    dfs_indices[np.isnan(dfs_indices)] = 0
    return dfs_values, dfs_indices



def create_autoencoder_training_data(dfs_values, neighbor_detectors_dict, detector_list):
    def get_detector_timeseries(detector_id):
        if np.isnan(detector_id):
            return np.zeros(288)
        else:
            idx = detector_row_idx_dict[detector_id]
            return dfs_values[i, idx]
    
    #create training for autoencoder
    X_train_autoencoder = []
    y_train_autoencoder = []

    detector_list.sort()
    detector_row_idx_dict = {elem: index for index, elem in enumerate(detector_list)}

    for i in range(84, len(dfs_values)):
        # compute historical dataset as 2 month same-day traffic condition
        hist_array = np.mean(dfs_values[[i-k*7 for k in range(12)], :, :], axis = 0)
        
        # X_train_list.append(np.concatenate([dfs_values[i], hist_array]))
        for detector_id in detector_list:
            upstream_detector_id, downstream_detector_id = neighbor_detectors_dict[detector_id]
            current_vector = get_detector_timeseries(detector_id)
            upstream_vector = get_detector_timeseries(upstream_detector_id)
            downstream_vector = get_detector_timeseries(downstream_detector_id)

            history_vector = hist_array[detector_row_idx_dict[detector_id]]
            train_vector = np.concatenate([current_vector, upstream_vector, downstream_vector, history_vector])

            X_train_autoencoder.append(train_vector)
            y_train_autoencoder.append(current_vector)

    X_train_autoencoder = np.array(X_train_autoencoder)
    y_train_autoencoder = np.array(y_train_autoencoder)

    return X_train_autoencoder, y_train_autoencoder


def custom_train_val_split(n, n_detector, train_ratio=0.8):
    import random
    random.seed(2024)
    # training/val split for GAT 
    indices = list(range(n))
    random.shuffle(indices)
    train_size = int(train_ratio * n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # training/val split for autoencoder
    train_indices_autoencoder = []
    val_indices_autoencoder = []

    for train_index in train_indices:
        train_indices_autoencoder.extend(list(range(train_index*n_detector,(train_index+1)*n_detector)))

    for val_index in val_indices:
        val_indices_autoencoder.extend(list(range(val_index*n_detector,(val_index+1)*n_detector)))

    return train_indices, val_indices, train_indices_autoencoder, val_indices_autoencoder


def create_autoencoder_training_data_from_graph_data(X_graph, X_mask, dfs_values, dfs_indices, neighbor_detectors_dict, detector_list):
    print("----------------- Create training data for Autoencoder --------------------")
    def get_detector_timeseries(detector_id):
        if np.isnan(detector_id):
            return np.zeros(288)
        else:
            idx = detector_row_idx_dict[detector_id]
            return X_graph_observed[i, idx]
        
    num_data_copies = 9  # the tensor is copied for 9 times to construct errors
    hist_graph = []
    for i in range(84, len(dfs_values)):
        # compute historical dataset as 3 month same-day traffic condition
        hist_graph.append(np.mean(dfs_values[[i-k*7 for k in range(12)], :, :], axis = 0))
    hist_graph = np.array(hist_graph)

    X_graph_observed = X_graph*X_mask

    #create training for autoencoder
    X_train_autoencoder = []
    y_train_autoencoder = []

    detector_list.sort()
    # in dfs_values, the detectors are sorted
    detector_row_idx_dict = {elem: index for index, elem in enumerate(detector_list)}
    n_days_interval = len(range(84, len(dfs_values)))  # 191 in our case

    for i in range(X_graph.shape[0]):
        for detector_id in detector_list:
            upstream_detector_id, downstream_detector_id = neighbor_detectors_dict[detector_id]
            current_vector = get_detector_timeseries(detector_id)
            upstream_vector = get_detector_timeseries(upstream_detector_id)
            downstream_vector = get_detector_timeseries(downstream_detector_id)
            # index 192 input data share the historical average with index 1 data (%191)
            history_vector = hist_graph[i%n_days_interval, detector_row_idx_dict[detector_id]]

            train_vector = np.concatenate([current_vector, upstream_vector, downstream_vector, history_vector])

            X_train_autoencoder.append(train_vector)
            y_train_autoencoder.append(current_vector)
    return np.array(X_train_autoencoder), np.array(y_train_autoencoder)


    # for i in range(84, len(dfs_values)):
    #     # compute historical dataset as 3 month same-day traffic condition
    #     hist_array = np.mean(dfs_values[[i-k*7 for k in range(12)], :, :], axis = 0)
        
    #     # X_train_list.append(np.concatenate([dfs_values[i], hist_array]))
    #     for detector_id in detector_list:
    #         upstream_detector_id, downstream_detector_id = neighbor_detectors_dict[detector_id]
    #         current_vector = get_detector_timeseries(detector_id)
    #         upstream_vector = get_detector_timeseries(upstream_detector_id)
    #         downstream_vector = get_detector_timeseries(downstream_detector_id)
    #         history_vector = hist_array[detector_row_idx_dict[detector_id]]

    #         train_vector = np.concatenate([current_vector, upstream_vector, downstream_vector, history_vector])

    #         X_train_autoencoder.append(train_vector)
    #         y_train_autoencoder.append(current_vector)

    # X_train_autoencoder = np.array(X_train_autoencoder)
    # y_train_autoencoder = np.array(y_train_autoencoder)

    # return X_train_autoencoder, y_train_autoencoder

    

def create_graph_training_data(dfs_values, dfs_indices, imputed_as_ground_truth=True): 
    print("----------------- Create training data for GAT --------------------")
    import random
    random.seed(2024)   
    def add_random_missing(X_graph, X_mask):
        '''
        This function adds random missing to the data tensor by a random ratio p
        '''
        print("------------- Adding random missing to data ...")
        for p in [0.3, 0.4, 0.5]:
            random_missing_mask = np.random.choice([0, 1], size=tensor_graph.shape, p=[p, 1-p])
            # use original data if not missing, use historical data if it is missing 
            filled_input_tensor = random_missing_mask*tensor_graph + (1-random_missing_mask)*hist_graph
            X_graph.extend(list(filled_input_tensor))
            X_mask.extend(list(random_missing_mask*mask_graph))
        return X_graph, X_mask

    def add_block_missing(X_graph, X_mask):
        '''
        This function adds block missing to the data tensor
        '''
        print("------------- Adding block missing to data ...")
        shape = tensor_graph[0].shape
        n_rows, n_cols = shape
        
        for _iter in range(5):
            for k in range(len(tensor_graph)):
                block_missing_mask = np.ones(shape)
                # for each day, implement block missing
                # create a block_missing_mask for each day's 2D array
                for i in range(n_rows):
                    # generate continuous missing 
                    missing_length = np.random.randint((_iter+1)*20, 288)
                    start_col = np.random.randint(0, n_cols - missing_length + 1)
                    block_missing_mask[i, start_col:start_col + missing_length] = 0
                
                block_missing_mask = block_missing_mask*mask_graph[k]
                filled_input_array = block_missing_mask*tensor_graph[k] + (1-block_missing_mask)*hist_graph[k]
                X_graph.append(filled_input_array)
                X_mask.append(block_missing_mask)
        return X_graph, X_mask
    
    X_graph = []
    X_mask = []
                
    tensor_graph = dfs_values[84:]
    # create a tensor for historial average, it has the same dim as tensor_graph
    hist_graph = []
    for i in range(84, len(dfs_values)):
        # compute historical dataset as 3 month same-day traffic condition
        hist_graph.append(np.mean(dfs_values[[i-k*7 for k in range(12)], :, :], axis = 0))
    hist_graph = np.array(hist_graph)

    if imputed_as_ground_truth==True:
        # we consider the current imputed data by Caltrans as ground truth
        # this is considered to avoid the situation that some detector never had ground truth
        mask_graph = np.ones_like(tensor_graph)
    else: 
        # the current imputed data is not considered as ground truth
        # these imputed data should not be used to penalize in the loss function
        mask_graph = dfs_indices[84:]
        tensor_graph = mask_graph*tensor_graph + (1-mask_graph)*hist_graph

    # now we get tensor_graph, mask_graph, hist_graph, all having the same dimension
    X_graph.extend(list(tensor_graph))
    X_mask.extend(list(mask_graph))  
    X_graph, X_mask = add_random_missing(X_graph, X_mask)
    X_graph, X_mask = add_block_missing(X_graph, X_mask)
    print("Finished")
    return np.array(X_graph), np.array(X_mask)



def GAT_adjacency_matrix(detector_list, detector_to_station_dict, station_to_detector_dict, neighbor_stations_dict, neighbor_detectors_dict, consider_two_neighbors=True):
    # construct Adjacency matrix
    n = len(detector_list)
    A = np.zeros((n, n))

    for detector_id in detector_list:
        row_idx = detector_list.index(detector_id)
        if consider_two_neighbors==False:  
            # consider all nearby detectors as neighbors  
            connected_detectors = []
            station_id = detector_to_station_dict[detector_id]
            upstream_station_id, downstream_station_id = neighbor_stations_dict[station_id][0]
            if not np.isnan(upstream_station_id):
                connected_detectors.extend(station_to_detector_dict[upstream_station_id])
            if not np.isnan(downstream_station_id):
                connected_detectors.extend(station_to_detector_dict[downstream_station_id])
                
            # Retrieve indices of elements from l2 in l1
            adj_idx = [i for i, x in enumerate(detector_list) if x in connected_detectors]
            A[row_idx, adj_idx] = 1
        else: 
            # simply consider two neighbors
            adj_idx = [i for i, x in enumerate(detector_list) if x in neighbor_detectors_dict[detector_id]]
            A[row_idx, adj_idx] = 1

    A = A+np.eye(len(A))
    return A