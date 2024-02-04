# =============================================================================
# gabor filters
# =============================================================================

import cv2
import numpy as np

def apply_gabor_filter(image, kernel_size=10, sigma=3.0, theta=0, lamda=10.0, gamma=0.8, psi=0):
    
    # generate 11 by 11 Gabor kernel (just a matrix based on parameteres)
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
    
    # filter image
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return filtered_image

def gabor_features(image):
    
    image = image.reshape((16, 15))
    
    # this can vary, simple choice for now
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    features = []

    for theta in orientations:
        
        # convolute image using gabor filter
        filtered_image = apply_gabor_filter(image, theta=theta)
        
        # just extracting the mean and variance for now
        # maybe more elaborate later
        features.extend([np.mean(filtered_image), np.std(filtered_image)])

    return np.round(features, 3)

if __name__ == "__main__":

    # load data
    with open('../data.txt', 'r') as file:
        dataset = np.array([list(map(int, line.split())) for line in file])

    # partition data into training and testing datasets
    train_data = np.empty((0, 240))
    test_data  = np.empty((0, 240))
    for i in range(10):
        train_data = np.vstack((train_data, dataset[200 * i : 200 * i + 100]))
        test_data  = np.vstack((test_data , dataset[200 * i + 100 : 200 * i + 200]))
        
    image = test_data[200]
    features = gabor_features(image)
    # print(features)
    