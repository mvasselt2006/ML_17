# =============================================================================
# computes the horizontal and vertical symmetry of a given image
# =============================================================================

import numpy as np

# flip first 7 columns and compute sum of differences between flipped and final
# 8 columns
def vert_symmetry(image):

    image = image.reshape((16, 15))

    # split image into left and right halves
    left_half = image[:, :7]
    right_half = image[:, 8:]
    
    flipped_right_half = np.fliplr(right_half)
    difference = np.abs(left_half - flipped_right_half)
    symmetry_score = np.sum(difference)

    return symmetry_score / 100

# flip first 8 rows and compute sum of differences between flipped and final 8
# rows
def hori_symmetry(image):

    image = image.reshape((16, 15))

    # split image into top and bottom halves
    top_half = image[:8, :]
    bottom_half = image[8:, :]

    flipped_bottom_half = np.fliplr(bottom_half)
    difference = np.abs(top_half - flipped_bottom_half)
    symmetry_score = np.sum(difference)

    return symmetry_score / 100

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
    
    averages = []
    for i in range(10):
        avg_sym_score = 0
        for j in range(100):
            avg_sym_score += vert_symmetry(train_data[100 * i + j])
        avg_sym_score /= 100
        averages.append([i, round(avg_sym_score, 2)])
    
    for (digit, avg_sym_score) in sorted(averages, key = lambda x : x[1], reverse = False):
        print(f"average vertical symmetry score of {digit}: {avg_sym_score}")
    print()
    
    averages = []
    for i in range(10):
        avg_sym_score = 0
        for j in range(100):
            avg_sym_score += hori_symmetry(train_data[100 * i + j])
        avg_sym_score /= 100
        averages.append([i, round(avg_sym_score, 2)])
    
    for (digit, avg_sym_score) in sorted(averages, key = lambda x : x[1], reverse = False):
        print(f"average horizontal symmetry score of {digit}: {avg_sym_score}")
