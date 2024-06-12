import numpy as np

def baseline_history_imputation_mse(dfs_values, X_val, mask_val, val_indices):
    import torch
    '''
    Use the 3 month historical measurements as the baseline for comparison
    '''
    hist_graph = []
    for i in range(84, len(dfs_values)):
        # compute historical dataset as 3 month same-day traffic condition
        hist_graph.append(np.mean(dfs_values[[i-k*7 for k in range(12)], :, :], axis = 0))
    hist_graph = np.array(hist_graph)
    
    n_days_interval = len(range(84, len(dfs_values)))  # 191 in our case
    
    if isinstance(X_val, torch.Tensor):
        X_val = X_val.numpy()
    if isinstance(mask_val, torch.Tensor):
        mask_val = mask_val.numpy()

    num_mask_element = 0
    squared_error = 0
    for i in range(len(X_val)):
        day_index = val_indices[i]
        hist_graph_i = hist_graph[day_index%n_days_interval]
        squared_error += np.sum((hist_graph_i - X_val[i])**2 * mask_val[i])
        num_mask_element += np.sum(mask_val[i])
    mse = squared_error/num_mask_element
    print("The MSE of a naive 3-month historical average imputation is", mse)
    return mse


def plot_estimation_history(i, x_train, y_train, y_predict):
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [1, 9]})  # Adjusted the figure height and height ratios
    
    # Plot using imshow
    matrix_colors = np.zeros(x_train[i].shape)
    matrix_colors[x_train[i]!=0] = 1
    matrix_colors = matrix_colors.reshape(1,-1)
    axs[0].imshow(matrix_colors, cmap='binary', aspect='auto', vmin=0, vmax=1)
        # Add boundary around the plot
    rect = plt.Rectangle((-0.5, -0.5), matrix_colors.shape[1], 1, linewidth=1, edgecolor='black', facecolor='none')
    axs[0].add_patch(rect)

    # Set plot limits and remove axes
    axs[0].set_xlim(0, 1152)
    axs[0].set_ylim(-0.5, 1)  # Adjusted the y-axis limit to ensure the boundary line is visible
    axs[0].axis('off')
    axs[0].axes.get_yaxis().set_visible(False)
#     axs[0].set_aspect(50)

    # Plot using plt.plot
    axs[1].plot(y_predict[i], label='Imputed')
    # axs[1].plot(0, label=' ')

    axs[1].plot(y_train[i], label='Ground truth')
    # plot historical data too
    axs[1].plot(x_train[i][-288:], label='3-month historical data')
    # axs[1].plot(x_train[i][288:2*288], label='Upstream')
#     axs[1].plot(x_train[i][2*288:3*288], label='Downstream')
    
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Occupancy')
    axs[1].legend()




# def plot_estimation_graph(i, x_train, x_mask, y_predict):
#     import matplotlib.pyplot as plt
    
#     fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [1, 9]})  # Adjusted the figure height and height ratios
    
#     # Plot using imshow
#     matrix_colors = np.zeros(x_mask[i].shape)
#     matrix_colors[x_mask[i]!=0] = 1
#     matrix_colors = matrix_colors.reshape(1,-1)
#     axs[0].imshow(matrix_colors, cmap='binary', aspect='auto', vmin=0, vmax=1)
#         # Add boundary around the plot
#     rect = plt.Rectangle((-0.5, -0.5), matrix_colors.shape[1], 1, linewidth=1, edgecolor='black', facecolor='none')
#     axs[0].add_patch(rect)

#     # Set plot limits and remove axes
#     axs[0].set_xlim(0, 288)
#     axs[0].set_ylim(-0.5, 1)  # Adjusted the y-axis limit to ensure the boundary line is visible
#     axs[0].axis('off')
#     axs[0].axes.get_yaxis().set_visible(False)
# #     axs[0].set_aspect(50)

#     # Plot using plt.plot
#     axs[1].plot(y_predict[i], label='Imputed')
#     # axs[1].plot(0, label=' ')

#     axs[1].plot(x_train[i], label='Ground truth')

#     axs[1].set_xlabel('Time')
#     axs[1].set_ylabel('Occupancy')
#     axs[1].legend()



def plot_estimation_graph(i, x_truth, x_mask, y_predict):
    import matplotlib.pyplot as plt

    def find_segments(matrix_colors):
        segments = []
        start = 0
        while start < len(matrix_colors):
            end = start
            while end < len(matrix_colors) and matrix_colors[end] == matrix_colors[start]:
                end += 1
            segments.append((start, end, matrix_colors[start]))
            start = end
        return segments

    fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [1, 9]})  # Adjusted the figure height and height ratios
    
    # Prepare the matrix_colors based on x_mask
    matrix_colors = np.zeros(x_mask[i].shape)
    matrix_colors[x_mask[i] != 0] = 1
    matrix_colors = matrix_colors.reshape(-1)  # Ensure it's a 1D array

    segments = find_segments(matrix_colors)

    # Plot the matrix_colors using imshow
    axs[0].imshow(matrix_colors.reshape(1, -1), cmap='binary', aspect='auto', vmin=0, vmax=1)
    
    # Add boundary around the plot
    rect = plt.Rectangle((-0.5, -0.5), matrix_colors.shape[0], 1, linewidth=1, edgecolor='black', facecolor='none')
    axs[0].add_patch(rect)

    # Set plot limits and remove axes
    axs[0].set_xlim(0, 288)
    axs[0].set_ylim(-0.5, 1)  # Adjusted the y-axis limit to ensure the boundary line is visible
    axs[0].axis('off')
    axs[0].axes.get_yaxis().set_visible(False)

    # Plot the imputed values
    axs[1].plot(y_predict[i], label='GAT Imputed')

    # Plot the ground truth with solid and dashed lines based on matrix_colors
    for start, end, color in segments:
        linestyle = '-' if color == 1 else '--'
        axs[1].plot(range(start, end), x_truth[i][start:end], linestyle=linestyle, color='orange', label='Caltrans Imputed (Ground Truth)' if start == 0 else "")

    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Occupancy')
    axs[1].legend()

    plt.show()