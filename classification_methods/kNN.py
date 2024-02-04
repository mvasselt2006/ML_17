#!/usr/bin/python3
# =============================================================================
# evaluates kNN on the data sets feature, raw and combined
# evaluates different kNN (inspired) options
# =============================================================================

from sklearn.neighbors import *
from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_decision_regions

import numpy as np
import matplotlib.pyplot as plt

def give_knn(n=5, w="uniform", p=2):
    return lambda _ : (KNeighborsClassifier(n_neighbors = n, weights = w, p=p),
                       f"kNN k: {n} distance: {w} p: {p}")

def give_nearest_centroid():
    return lambda _ : (NearestCentroid(metric="euclidean"),
                       f"nearest centroid")

def assess_pred(data_form, model):
    
    model, text = model(0)

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

    print(f"data: {data_form} {text} misclassified: {misclassifications}/{datalen}")

datasets = ["raw", "feats", "raw_and_feats"]
distances = ["uniform", "distance"]

for dataset in datasets:
    assess_pred(dataset, give_nearest_centroid())
    for n_neighbors in range(3,9):
        for distance in distances:
            for p in [2]:
                assess_pred(dataset, give_knn(n_neighbors, distance, p))

# just trying NCA out too, mostly to generate plots

def assess_nca(data_form):

    n = 2 # for 2D plot

    nca = NeighborhoodComponentsAnalysis(n_components = n, init="pca", 
                                           max_iter=50, tol=1e-5) # convergence iterations and tolerance

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

    nca.fit(X_scaled, y_train)
    X_nca = nca.transform(X_scaled)
    X_nca_test = nca.transform(X_test_scaled)

    # only for graphing the decision plot_decision_regions
    # todo problem: this graphs NCA decision regions = kNN trained on dimension
    # reduced data, not the original kNN model
    knn = KNeighborsClassifier(n_neighbors = 5, weights = "distance", p=2)
    knn.fit(X_nca, y_train)

    # generate scatter plot for test data
    
    plt.title(f"Distribution of digits in the top two NCA dimensions, data {dataset}")
#    for i in range(10):
#        x_selected = X_nca_test[y_test == i]
#        plt.scatter(x_selected[:, 0], x_selected[:, 1], label=str(i))
    plot_decision_regions(X_nca_test, y_test.astype(np.int_), knn)
    plt.legend()
    plt.show()
    

for dataset in datasets: 
    assess_nca(dataset)
