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


from sklearn.mixture import GaussianMixture

# Create and fit the model
gmm = GaussianMixture(n_components=5)
gmm.fit(train_data)