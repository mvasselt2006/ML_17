# =============================================================================
# compute sum of pixel values in a given row or column
# =============================================================================

import numpy as np

# counts the number of non-zero values in a column
def col_count_sum(image_data, col_idx):
    return np.sum(image_data.reshape((16, 15))[:, col_idx])

# counts the number of non-zero values in a row
def row_count_sum(image_data, row_idx):
    return np.sum(image_data.reshape((16, 15))[row_idx, :])

if __name__ == "__main__":

    # load data
    with open('../data.txt', 'r') as file:
        dataset = np.array([list(map(int, line.split())) for line in file])

    # parition data into training and testing datasets
    train_data = np.empty((0, 240))
    test_data  = np.empty((0, 240))
    for i in range(10):
        train_data = np.vstack((train_data, dataset[200 * i : 200 * i + 100]))
        test_data  = np.vstack((test_data , dataset[200 * i + 100 : 200 * i + 200]))
    
    
    image = train_data[0]
    for col_idx in range(15):
        print(col_count_sum(image, col_idx))
        
    for row_idx in range(16):
        print(row_count_sum(image, row_idx))
        