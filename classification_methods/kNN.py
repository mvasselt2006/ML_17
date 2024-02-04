#!/usr/bin/python3
# =============================================================================
# evaluates kNN on the data sets feature, raw and combined
# evaluates different kNN (inspired) options
# 
# Usage: ./kNN.py while in the classification_methods subfolder
# Relies on the assumption that ../X_train.csv and ../X_test.csv exist for all X in {"feats", "raw", "raw_and_feats"} with up to date raw and feature data
# 
# =============================================================================

from sklearn.neighbors import *
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt

def give_knn(n=5, w="uniform", p=2):
    return lambda _ : (KNeighborsClassifier(n_neighbors = n, weights = w, p=p, algorithm="brute"),
                       f"kNN k: {n} distance: {w} p: {p}")

def weight_scale_features(data, weight):
    raw = data[:, :239]
    feats = data[:, 240:]
    feats = feats * weight
    data = np.concatenate((raw, feats), axis=1) 
    return data

def assess_pred(data_form, model, weight=1.0):
    # weight: if different than 1, after scaling it will multiply the values in columns beyond column 240 by the value

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

    if weight != 1.0:
        X_scaled = weight_scale_features(X_scaled, weight)
        X_test_scaled = weight_scale_features(X_test_scaled, weight)

    # train/fit model
    model.fit(X_scaled, y_train)

    # use model to predict on test data
    y_pred = model.predict(X_test_scaled)

    misclassifications = (y_pred != y_test).sum()
    datalen = len(y_test)

    error = (1 - misclassifications / datalen) * 100

    print(f"data: {data_form} w={weight} {text} error: {error}")
    return error

# k values
ks = np.arange(1, 10, 1)
# distance weighting
raw_y = []
feats_y = []
combined_y = []
# uniform weighting
uraw_y = []
ufeats_y = []
ucombined_y = []

for n_neighbors in ks:
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
plt.plot(ks, combined_y, "g-", label="R+HF data (distance)")
plt.plot(ks, ucombined_y, "g--", label="R+HF data (uniform)")
plt.plot(ks, feats_y, "y-", label="Handcrafted features data (distance)")
plt.plot(ks, ufeats_y, "y--", label="Handcrafted features data (uniform)")
plt.plot(ks, raw_y, "b-", label="Raw data (distance)")
plt.plot(ks, uraw_y, "b--", label="Raw data (uniform)")
plt.xlabel("Number of neighbors $k$")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# bonus / not in report / relative importance of features vs raw data in the feat + raw dataset is examined
weights = np.arange(0, 3.0, 0.1)
ys = []
for weight in weights:
    ys.append(assess_pred("raw_and_feats", give_knn(3, "distance", 2), weight))

plt.plot(weights, ys)

plt.legend()
plt.show()
    
