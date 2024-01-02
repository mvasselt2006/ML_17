# linear regression program
# taken from some website and modified

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class LinRegr(): 
    
    def __init__(self, learning_rate, iterations): 
        self.learning_rate = learning_rate 
        self.iterations = iterations 
    
    def fit(self, X, y): 
        
        # dataframe X is an N by p matrix
        # N samples, p features
        self.N, self.p = X.shape 
        
        # weight initialization
        self.weights = np.zeros(self.p) 
        self.bias = 0
        self.X = X 
        self.y = y 
        
        # gradient descent learning
        for i in range(self.iterations): 
            self.update_weights() 
            
        return self
    
    def update_weights(self): 
        y_pred = self.predict(self.X) 
        
        # compute gradients
        grad_weights = -(2 * (self.X.T).dot(self.y - y_pred)) / self.N
        grad_bias = -2 * np.sum(self.y - y_pred) / self.N 
        
        # update weights
        self.weights = self.weights - self.learning_rate * grad_weights 
        self.bias = self.bias - self.learning_rate * grad_bias 
        
        return self
    
    def predict(self, X): 
        return X.dot(self.weights) + self.bias

def main():

    # load data
    df = pd.read_csv("../data.csv")

    # normalise pixel values to be in [0, 1]
    df /= 6

    # define correct labels
    y_train = np.repeat(np.arange(10), 100)
    y_test = np.repeat(np.arange(10), 100)
    
    # partition data into training and testing datasets
    X_train = np.empty((0, 240))
    X_test  = np.empty((0, 240))
    for i in range(10):
        X_train = np.vstack((X_train, df[200 * i : 200 * i + 100]))
        X_test  = np.vstack((X_test , df[200 * i + 100 : 200 * i + 200]))

    # apply PCA to the train and test data
    pca = PCA(n_components = 0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # train model
    model = LinRegr(iterations = 1000, learning_rate = 0.1)
    model.fit(X_train, y_train)
    
    # predict on test set and round outcome to nearest integer
    y_pred = model.predict(X_test)
    y_pred = np.rint(y_pred).astype(int)

    print('accuracy:', np.round(100 * np.mean(y_pred == y_test), 2))

if __name__ == "__main__": 
    main()
