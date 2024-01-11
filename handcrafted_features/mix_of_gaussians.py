# =============================================================================
# unsure how to use for lin. regression/random forest as of 02/01/2024
# but i think this should be useful
# =============================================================================

from sklearn.mixture import GaussianMixture
import numpy as np

def train_MOGs(images, n_components=3):

    data = np.vstack([image.flatten() for image in images])
    MOG = GaussianMixture(n_components=n_components, random_state=0)
    MOG.fit(data)

    return MOG

def train_individual(train_data):

    # train MOGs for each digit
    MOG_0 = train_MOGs(train_data[0:100])
    MOG_1 = train_MOGs(train_data[100:200])
    MOG_2 = train_MOGs(train_data[200:300])
    MOG_3 = train_MOGs(train_data[300:400])
    MOG_4 = train_MOGs(train_data[400:500])
    MOG_5 = train_MOGs(train_data[500:600])
    MOG_6 = train_MOGs(train_data[600:700])
    MOG_7 = train_MOGs(train_data[700:800])
    MOG_8 = train_MOGs(train_data[800:900])
    MOG_9 = train_MOGs(train_data[900:1000])
    
    MOGs = [MOG_0, MOG_1, MOG_2, MOG_3, MOG_4,
            MOG_5, MOG_6, MOG_7, MOG_8, MOG_9]
    
    return MOGs

def classify(image, MOGs):

    image = image.flatten()
    image = image.reshape(1, -1)

    log_likelihoods = []
    for MOG in MOGs:
        log_likelihoods.append(MOG.score(image))
    classification = np.argmax(log_likelihoods)

    return classification

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
    
    MOGs = train_individual(train_data)
    
    for i in range(10):
        count = 0
        for j in range(100):
            image = test_data[100 * i + j]
            count += (classify(image, MOGs) != i)
        print(f"correct classification rate for digit {i}:", (100 - count) / 100)
