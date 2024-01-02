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

# map all 6s to 0s and non-6s to 1s
# this is the method that the example paper took
transformed_data = np.where(train_data == 6, 0, 1)

# returns the vert ratio of a ingle image (input as row vector)
# follows the formula used in the example paper
def vert_ratio_image(image_data, k):
    
    image_data = image_data.reshape((16, 15))
    
    ratio_numerator = 0
    for i in range(k):
        for j in range(15):
            ratio_numerator += image_data[i][j]
    
    return ratio_numerator / image_data.sum()

# returns the average vertical ratio of a set of images
def avg_vert_ratio(data, k):
    
    avg = 0
    for i in range(len(data)):
        avg += vert_ratio_image(data[i], k)
    avg /= len(data)
    
    return avg

k = 8
for i in range(10):    
    print(i, np.round(avg_vert_ratio(transformed_data[100 * i : 100 * i + 100], k), 3))
