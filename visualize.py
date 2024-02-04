#!/usr/bin/python3
# =============================================================================
# The goal of this is to visualize the features for the report
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

train_df = np.loadtxt("feats_train.csv", delimiter = ",", skiprows=1)
test_df = np.loadtxt("feats_test.csv", delimiter = ",", skiprows=1)

X_train = train_df[:, :-1]
y_train = train_df[:,  -1]
    
X_test = test_df[:, :-1]
y_test = test_df[:,  -1]

# sim. with 0,sim. with 1,sim. with 2,sim. with 3,sim. with 4,sim. with 5,sim. with 6,sim. with 7,sim. with 8,sim. with 9,hole_area,vert_ratio,col_count_0,col_count_1,col_count_2,col_count_3,col_count_4,col_count_5,col_count_6,col_count_7,col_count_8,col_count_9,col_count_10,col_count_11,col_count_12,col_count_13,col_count_14,row_count_0,row_count_1,row_count_2,row_count_3,row_count_4,row_count_5,row_count_6,row_count_7,row_count_8,row_count_9,row_count_10,row_count_11,row_count_12,row_count_13,row_count_14,row_count_15,col_sum_0,col_sum_1,col_sum_2,col_sum_3,col_sum_4,col_sum_5,col_sum_6,col_sum_7,col_sum_8,col_sum_9,col_sum_10,col_sum_11,col_sum_12,col_sum_13,col_sum_14,row_sum_0,row_sum_1,row_sum_2,row_sum_3,row_sum_4,row_sum_5,row_sum_6,row_sum_7,row_sum_8,row_sum_9,row_sum_10,row_sum_11,row_sum_12,row_sum_13,row_sum_14,row_sum_15,gabor_0,gabor_1,gabor_2,gabor_3,gabor_4,gabor_5,gabor_6,gabor_7,hog_descriptor_0,hog_descriptor_1,hog_descriptor_2,hog_descriptor_3,hog_descriptor_4,hog_descriptor_5,hog_descriptor_6,hog_descriptor_7,hog_descriptor_8,hog_descriptor_9,hog_descriptor_10,hog_descriptor_11,hog_descriptor_12,hog_descriptor_13,hog_descriptor_14,hog_descriptor_15,hog_descriptor_16,hog_descriptor_17,hog_descriptor_18,hog_descriptor_19,hog_descriptor_20,hog_descriptor_21,hog_descriptor_22,hog_descriptor_23,hog_descriptor_24,hog_descriptor_25,hog_descriptor_26,hog_descriptor_27,hog_descriptor_28,hog_descriptor_29,hog_descriptor_30,hog_descriptor_31,hog_descriptor_32,hog_descriptor_33,hog_descriptor_34,hog_descriptor_35,label

columns = {}

with open("feats_train.csv") as f:
    title_line = f.readline().strip("\n")
    column_list = title_line.split(",")
    for i in range(len(column_list)):
        columns[column_list[i]] = i

# visualize similarity to 0 in a scatter plot
def scatter(name, featid):
    for i in range(10):
        data = X_test[y_test == i]
        plt.scatter(data[:, columns[featid]], y_test[y_test == i], label=str(i))

    plt.xlabel(name)
    plt.ylabel("Digit")
    plt.legend()
    plt.show()

#scatter("Similarity with 0", "sim. with 0")
scatter("Gabor 2", "gabor_2")

# second test task: x axis: similarity with 0, y axis: how frequently it happens per class as a stackplot
# Kernel Density estimation for smoothness, so no ugly blocky histograms

def feature_stackplot(name, featid, bandwidth=0.02, kernel_type="gaussian"):
    kde = KernelDensity(kernel = kernel_type, bandwidth=bandwidth)

    pdfs = []

    featdata = X_test[:, columns[featid]]
    x = np.linspace(np.min(featdata), np.max(featdata), 1000)
    labels = []

    for i in range(10):
        data = featdata[y_test == i]
        kde.fit(data[:, None])
        pdfs.append(np.exp(kde.score_samples(x[:, None])) / len(data))
        labels.append(str(i))

    plt.stackplot(x, np.vstack(pdfs), labels=labels)
    plt.xlabel(name)
    plt.ylabel("Probability")
    plt.yticks([])
    plt.legend()
    plt.show()

#feature_stackplot("Similarity with 0", "sim. with 0", 0.02, "gaussian")
#feature_stackplot("Hole area", "hole_area", 5, "gaussian")
#feature_stackplot("Column count 8", "col_count_8", 1, "gaussian")
#feature_stackplot("Column sum 8", "col_sum_8", 5, "gaussian")
#feature_stackplot("Vertical ratio", "vert_ratio", 0.01, "gaussian")
#feature_stackplot("HOG 11", "hog_descriptor_11", 0.005, "gaussian")
feature_stackplot("Gabor 2", "gabor_2", 1, "gaussian")
