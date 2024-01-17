import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import os

def OurRandomForest(data, labels, criterion, train_size, random_seed, n_trees):
    # Splits the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, random_state=random_seed)

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, criterion= criterion, random_state=random_seed)

    # Train the classifier
    rf_classifier = rf_classifier.fit(X_train, y_train)

    # Predict on the test set
    predictions = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

random_seed = None

data = np.loadtxt('data.txt')
labels = np.zeros(2000)
for i in range(10):
    labels[i*200:i*200+200] = i

sizes = np.arange(0.1, .6, .1)
trees = np.arange(50, 200, 1)
crits = ['gini', 'entropy', 'log_loss']
for j in range(5):
    print(f'{j}-th run:')
    results = []
    multi_score = []
    for i in range(len(crits)):
        #print(f'Evaluating criterion {crits[i]}...')
        for size in sizes:
            #print(f'Evaluating size {size}...')
            for tree in trees:
                multi_score.append(OurRandomForest(data, labels, crits[i], size, random_seed, tree))
                results.append((crits[i],size, tree, multi_score[-1]))

    best_results = max(results, key=lambda x : x[3])

    print('\nMaximum accuracy for multiscore:')
    print(f'criterion: \t{best_results[0]},\nset size: \t{best_results[1]:.2},\nn. of trees: \t{best_results[2]},\nAccuracy: \t{best_results[3]:.3}')

