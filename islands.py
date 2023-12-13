from proto_matching import reconstruct_image

import numpy as np

# load data
with open('data.txt', 'r') as file:
    dataset = np.array([list(map(int, line.split())) for line in file])

# parition data into training and testing datasets
train_data = np.empty((0, 240))
test_data  = np.empty((0, 240))
for i in range(10):
    train_data = np.vstack((train_data, dataset[200 * i : 200 * i + 100]))
    test_data  = np.vstack((test_data , dataset[200 * i + 100 : 200 * i + 200]))

# binarise data, 0s mapped to 1, non-0s mapped to 0
binarised_data = np.where(train_data == 0, 1, 0)

def horizontal_switches(image_data, row):
    
    image_data = image_data.reshape((16, 15))
    
    current_colour = image_data[row][0]
    num_of_changes = 0
    for i in range(15):
        if image_data[row][i] != current_colour:
            num_of_changes += 1
            current_colour = image_data[row][i]
    
    return num_of_changes // 2

def vertical_switches(image_data, col):
    
    image_data = image_data.reshape((16, 15))
    
    current_colour = image_data[0][col]
    num_of_changes = 0
    for i in range(16):
        if image_data[i][col] != current_colour:
            num_of_changes += 1
            current_colour = image_data[i][col]
    
    return num_of_changes // 2

n = 3
reconstruct_image(binarised_data[100 * n])
print()
print(horizontal_switches(binarised_data[100 * n], 8))
print(vertical_switches(binarised_data[100 * n], 8))