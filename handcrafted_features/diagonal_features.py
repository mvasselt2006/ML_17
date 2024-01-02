# =============================================================================
# unsure how useful this feature is as of 02/01/2024
# =============================================================================

import numpy as np

# load data
with open('../data.txt', 'r') as file:
    dataset = np.array([list(map(int, line.split())) for line in file])

# parition data into training and testing datasets
train_data = np.empty((0, 240))
test_data  = np.empty((0, 240))
for i in range(10):
    train_data = np.vstack((train_data, dataset[200 * i : 200 * i + 100]))
    test_data  = np.vstack((test_data , dataset[200 * i + 100 : 200 * i + 200]))

def diag_features(image):

    image = image.reshape((16, 15))
    
    features = {}
    for offset in range(-1, 2):
        diagonal = np.diag(image, k=offset)
        features[f'sum_diag_{offset}'] = np.sum(diagonal)
        features[f'nonzero_count_diag_{offset}'] = np.count_nonzero(diagonal)
    
    return features

def anti_diag_features(image):
    
    flipped_image = np.fliplr(image)
    
    features = {}
    for offset in range(-1, 2):
        diagonal = np.diag(flipped_image, k=offset)
        features[f'sum_antidiag_{offset}'] = np.sum(diagonal)
        features[f'nonzero_count_antidiag_{offset}'] = np.count_nonzero(diagonal)

    return features

if __name__ == "__main__":
    
    image = dataset[0]
    diag = diag_features(image)
    for i in diag:
        print(f"{i}: {diag[i]}")
