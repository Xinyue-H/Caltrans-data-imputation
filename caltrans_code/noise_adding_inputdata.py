import numpy as np

def add_noise(X_train, y_train):
    # import random
    import copy
    # random.seed(10)

    # # quick fix on nan values
    # X_train[np.isnan(X_train)] = 0
    # y_train[np.isnan(y_train)] = 0
    
    # add complete missing for current
    sample_num = len(X_train)
   
    X_train_new = copy.copy(X_train[0:sample_num])
    y_train_new = copy.copy(y_train[0:sample_num])
    X_train_new[:,0:288] = 0
    X_train = np.vstack([X_train, X_train_new])
    y_train = np.vstack([y_train, y_train_new])
    print("FINISH1")
    # # add complete missing for upstream
    # X_train_new = copy.copy(original_X_train)
    # X_train_new[:,288:288*2] = 0
    # X_train = np.vstack([X_train, X_train_new])
    # y_train = np.vstack([y_train, y_train_new])
    # print("FINISH2")

    # # add complete missing for downstream
    # X_train_new = copy.copy(original_X_train)
    # X_train_new[:,288*2:288*3] = 0
    # X_train = np.vstack([X_train, X_train_new])
    # y_train = np.vstack([y_train, y_train_new])
    # print("FINISH3")

    p_list = [0.5]

    for p in p_list: 
        X_mask = np.concatenate([1- (np.random.rand(sample_num, 288*3) < p), np.ones((sample_num, 288))], axis=1)
        X_train_new = X_train[0:sample_num] * X_mask
        X_train = np.vstack([X_train, X_train_new]) 
        y_train = np.vstack([y_train, y_train_new])
    return X_train, y_train