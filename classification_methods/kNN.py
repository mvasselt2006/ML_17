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
    return lambda _ : (KNeighborsClassifier(n_neighbors = n, weights = w, p=p, algorithm="brute"),
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

    error = (1 - misclassifications / datalen) * 100

    print(f"data: {data_form} {text} error: {error}")
    return error

datasets = ["raw", "feats", "raw_and_feats"]

n_x = np.arange(1, 10, 1)
raw_y = []
feats_y = []
combined_y = []
# uniform
uraw_y = []
ufeats_y = []
ucombined_y = []

for n_neighbors in n_x:
    # p didn't have a large effect on accuracy but it was slow when p > 2
    # distance as weight function is superior to uniform
    w = "distance"
    raw_y.append(assess_pred("raw", give_knn(n_neighbors, w, 2)))
    feats_y.append(assess_pred("feats", give_knn(n_neighbors, w, 2)))
    combined_y.append(assess_pred("raw_and_feats", give_knn(n_neighbors, w, 2)))
    w = "uniform" 
    uraw_y.append(assess_pred("raw", give_knn(n_neighbors, w, 2)))
    ufeats_y.append(assess_pred("feats", give_knn(n_neighbors, w, 2)))
    ucombined_y.append(assess_pred("raw_and_feats", give_knn(n_neighbors, w, 2)))
    
plt.title(f"Finding the best k to use in kNNs")
plt.plot(n_x, combined_y, "g-", label="R+HF data (distance)")
plt.plot(n_x, ucombined_y, "g--", label="R+HF data (uniform)")
plt.plot(n_x, feats_y, "y-", label="Handcrafted features data (distance)")
plt.plot(n_x, ufeats_y, "y--", label="Handcrafted features data (uniform)")
plt.plot(n_x, raw_y, "b-", label="Raw data (distance)")
plt.plot(n_x, uraw_y, "b--", label="Raw data (uniform)")
plt.xlabel("k")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

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
