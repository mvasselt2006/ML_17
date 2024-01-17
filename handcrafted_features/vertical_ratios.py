# =============================================================================
# computes the ratio of the sum of pixel values in rows 1, ..., k to the total
# number of pixels
# 
# iterating through k = 1, ..., 16 yields a best case of k = 8
# =============================================================================

import numpy as np

# returns the vert ratio of a single image (input as row vector)
# follows the formula used in the example paper
def vert_ratio_image(image, k):
    
    reshaped_image = image.reshape((16, 15))
    
    ratio_numerator = 0
    for i in range(k):
        ratio_numerator += sum(reshaped_image[i, :])
    
    return round(ratio_numerator / reshaped_image.sum(), 3)

# returns the average vertical ratio of a set of images
# used just to check some 
def avg_vert_ratio(data, k):
    
    avg = 0
    for i in range(len(data)):
        avg += vert_ratio_image(data[i], k)
    avg /= len(data)
    
    return avg

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

    # map all 6s to 0s and non-6s to 1s
    # this is the method that the example paper took
    transformed_data = np.where(train_data == 6, 0, 1)

    k = 8
    for i in range(10):    
        print(i, np.round(avg_vert_ratio(transformed_data[100 * i : 100 * i + 100], k), 3))
