# =============================================================================
# height to width ratio of box that bounds the image
# seems to not seperate the digits must, could be useless
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

def bound_box_ratio(image):
    
    image = image.reshape((16, 15))
    image = np.uint8(image)

    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0

    # bounding box of the largest contour
    x, y, width, height = cv2.boundingRect(contours[0])

    # ratio
    if width != 0:
        ratio = height / width
    else:
        0

    return round(ratio, 3)

if __name__ == "__main__":
    
    averages = []
    for i in range(10):
        avg_box_ratio = 0
        for j in range(100):
            avg_box_ratio += bound_box_ratio(train_data[100 * i + j])
        avg_box_ratio /= 100
        averages.append([i, round(avg_box_ratio, 2)])
    
    for (digit, avg_box_ratio) in sorted(averages, key = lambda x : x[1], reverse = True):
        print(f"average bounding box ratio of {digit}: {avg_box_ratio}")