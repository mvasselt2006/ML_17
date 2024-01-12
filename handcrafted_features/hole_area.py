# =============================================================================
# calculates the area of any apparent hole in an image
# seems to distinguish them pretty well
# =============================================================================

import numpy as np
import cv2

def hole_area(image):
    
    image = image.reshape((16, 15))
    image = np.uint8(image)

    # Threshold the image to get a binary image
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the area of holes
    hole_area = 0
    for i, h in enumerate(hierarchy[0]):
        if h[3] != -1:  # Has parent
            hole_area += cv2.contourArea(contours[i])

    return round(hole_area, 2)

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
    
    for i in range(600, 700):
        image = test_data[i]
        print(hole_area(image))
    
    # averages = []
    # for i in range(10):
    #     avg_hole_area = 0
    #     for j in range(100):
    #         avg_hole_area += hole_area(train_data[100 * i + j])
    #     avg_hole_area /= 100
    #     averages.append([i, round(avg_hole_area, 2)])
    
    # for (digit, avg_hole_area) in sorted(averages, key = lambda x : x[1], reverse = True):
    #     print(f"average hole area of {digit}: {avg_hole_area}")
