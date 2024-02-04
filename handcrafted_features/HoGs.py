# =============================================================================
# histogram of oriented gradients
# =============================================================================

import cv2
import numpy as np

def HoG_descriptor(image):
    
    # reshape into matrix and pixel value type
    image = image.reshape((16, 15)).astype(np.uint8)
    
    # parameters
    cell_size  = (6, 6)
    block_size = (2, 2)
    nbins = 10
    
    # HoG size is based on size of image
    h, w = 16, 15
    
    # initialise HoG
    hog = cv2.HOGDescriptor(_winSize=(w // cell_size[1] * cell_size[1] ,
                                      h // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1]   ,
                                        block_size[0] * cell_size[0])  ,
                            _blockStride=(cell_size[1], cell_size[0])  ,
                            _cellSize=(cell_size[1], cell_size[0])     ,
                            _nbins=nbins)
    
    return np.round(hog.compute(image), 3)

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
    print(HoG_descriptor(image))
    print(len(HoG_descriptor(image)))