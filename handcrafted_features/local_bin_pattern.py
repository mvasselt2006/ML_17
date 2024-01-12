import numpy as np
from skimage.feature import local_binary_pattern

def lbp_features(image, num_points=8, radius=1, num_bins=10):
    image = image.reshape((16, 15))
    lbp = local_binary_pattern(image, num_points, radius, method="uniform")
    
    # Calculate the histogram with specified bins
    (hist, _) = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Flatten the histogram to a 1D array
    return hist.flatten()

# Example usage
# Assuming 'image' is one of your 16x15 images represented as a flat array
# lbp_features = compute_lbp_features(image)


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
    print(lbp_features(image))