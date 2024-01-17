# =============================================================================
# gabor filters
# =============================================================================

import cv2
import numpy as np

def apply_gabor_filter(image, kernel_size=10, sigma=2.0, theta=0, lambd=10.0, gamma=0.5, psi=0):
    
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return filtered_image

def gabor_features(image):
    
    image = image.reshape((16, 15))
    
    # this can vary
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    features = []

    for theta in orientations:
        filtered_image = apply_gabor_filter(image, theta=theta)
        
        # just extracting the mean and variance for now
        features.extend([np.mean(filtered_image), np.var(filtered_image)])

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
        
    image = test_data[150]
    features = gabor_features(image)
    print(features)
    