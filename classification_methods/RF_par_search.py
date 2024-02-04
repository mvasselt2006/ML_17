# =============================================================================
# prints and plots the results of the RF classification from RF_evaluate.py
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Load the results
data = np.loadtxt('search_results.txt', dtype=str, delimiter=',')

# Transforms strings to floats
numbers = np.array([[col[-3], col[-2], col[-1]] for col in data], dtype = float)

# RAW print
maxraw = np.max(numbers[:,0])
maxraw_indices = np.transpose(np.where(numbers[:,0] == maxraw))
maxraw_values = [[data[index[0],0], data[index[0],1], data[index[0],2]] for index in maxraw_indices]
print(f'RAW: Max accuracy = {maxraw:.3f} reached for (n_estimators, min_splits, max_features) = {maxraw_values}.')

# FEATS print
maxfeats = np.max(numbers[:,1])
maxfeats_indices = np.transpose(np.where(numbers[:,1] == maxfeats))
maxfeats_values = [[data[index[0],0], data[index[0],1], data[index[0],2]] for index in maxfeats_indices]
print(f'FEATS: Max accuracy = {maxfeats:.3f} reached for (n_estimators, min_splits, max_features) = {maxfeats_values},')

# RAW + FEATS print
maxrPf = np.max(numbers[:,2])
maxrPf_indices = np.transpose(np.where(numbers[:,2] == maxrPf))
maxrPf_values = [[data[index[0],0], data[index[0],1], data[index[0],2]] for index in maxrPf_indices]
print(f'RAW + FEATS: Max accuracy = {maxrPf:.3f} reached for (n_estimators, min_splits, max_features) = {maxrPf_values}.')

# Plot of the search for the best number of trees given best min_sample_split and max_features
# Ugly but works 
plt.plot(data[543:724,0], numbers[543:724,0], label='Raw Pixel')
plt.plot(data[543:724,0], numbers[543:724,1], label='Handcrafted features')
plt.plot(data[1087:1267,0], numbers[1087:1267,2], label='RP + HF')
plt.xticks(data[:181,0][0::20])
plt.xlabel('Number of trees')
plt.ylabel('Accuracy score')
plt.title('Search for best number of trees\n')
plt.grid()
plt.legend()
plt.show()