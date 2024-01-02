# =============================================================================
# doesn't work as intended as of 02/01/2024 :(
# =============================================================================

import numpy as np
import cv2

# load data
with open('../data.txt', 'r') as file:
    dataset = np.array([list(map(int, line.split())) for line in file])

# parition data into training and testing datasets
train_data = np.empty((0, 240))
test_data  = np.empty((0, 240))
for i in range(10):
    train_data = np.vstack((train_data, dataset[200 * i : 200 * i + 100]))
    test_data  = np.vstack((test_data , dataset[200 * i + 100 : 200 * i + 200]))
    
def count_edges(image):

    image = image.reshape((16, 15))
    image = np.uint8(image)

    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    # Count the edge pixels
    edge_count = np.sum(edges != 0)

    return edge_count

dummy_image = np.random.randint(0, 255, (16, 15), dtype=np.uint8)

edge_count = count_edges(dummy_image)
print(f"Edge count: {edge_count}")

image = dataset[0]
print(count_edges(image))

# for i in range(1000):
#     image = dataset[i]
#     if count_edges(image) > 0:
#         print(i)