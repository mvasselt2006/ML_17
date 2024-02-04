#!/usr/bin/python3
# =============================================================================
# evaluates kNN on the data sets feature, raw and combined
# evaluates different kNN (inspired) options
# =============================================================================

from sklearn.neighbors import *
from sklearn.preprocessing import StandardScaler
import numpy as np

def assess_pred(data_form, n_neighbors = 5, weights = "uniform", p=2):
    
    model = KNeighborsClassifier(n_neighbors = n_neighbors, weights=weights, p=p)

    # load data frames
    train_df = np.loadtxt(f"../{data_form}_train.csv", delimiter = ",", skiprows=1)
    test_df = np.loadtxt(f"../{data_form}_test.csv", delimiter = ",", skiprows=1)

    # split training data into features and label
    X_train = train_df[:, :-1]
    y_train = train_df[:,  -1]
    
    # preprocessor to remove mean and normalize using the variance
    scaler = StandardScaler().fit(X_train)

    X_scaled = scaler.transform(X_train)

    # split testing data into features and label
    X_test = test_df[:, :-1]
    y_test = test_df[:,  -1]

    X_test_scaled = scaler.transform(X_test)

    # train/fit model
    model.fit(X_scaled, y_train)

    # use model to predict on test data
    y_pred = model.predict(X_test_scaled)

    misclassifications = (y_pred != y_test).sum()
    datalen = len(y_test)

    print(f"kNN data: {data_form} k: {n_neighbors} weights: {weights} p: {p} misclassified: {misclassifications}/{datalen}")

datasets = ["raw", "feats", "raw_and_feats"]
distances = ["uniform", "distance"]

for n_neighbors in range(3,9):
    for dataset in datasets:
        for distance in distances:
            for p in [2]:
                assess_pred(dataset, n_neighbors, distance, p)

