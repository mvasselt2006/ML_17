# =============================================================================
# generates feats_train.csv and feats_test.csv
# as of 02/01/2024, this uses just three features
# =============================================================================

# TODO: directory of data.txt has messed up slightly, need to fix

import numpy as np
import pandas as pd
import sys

# load data
with open('data.txt', 'r') as file:
    dataset = np.array([list(map(int, line.split())) for line in file])

# parition data into training and testing datasets
train_data = np.empty((0, 240))
test_data  = np.empty((0, 240))
for i in range(10):
    train_data = np.vstack((train_data, dataset[200 * i : 200 * i + 100]))
    test_data  = np.vstack((test_data , dataset[200 * i + 100 : 200 * i + 200]))
    
sys.path.append('./handcrafted_features')

from hole_area import hole_area
from bound_box_ratio import bound_box_ratio
from proto_matching import dot_prod_compare_remake

def get_data_frames():
    
    df_train = pd.DataFrame()
    df_test  = pd.DataFrame()
    for i in range(1000):
        
        train_image = train_data[i]
        test_image  = test_data[i]
        
        # hole area feature
        df_train.loc[i, 'hole_area'] = hole_area(train_image)
        df_test.loc[i, 'hole_area']  = hole_area(test_image)
        
        # bound box ratio feature
        df_train.loc[i, 'bound_box_ratio'] = bound_box_ratio(train_image)
        df_test.loc[i, 'bound_box_ratio']  = bound_box_ratio(test_image)
        
        # prototype matching featues
        for j in range(10):
            df_train.loc[i, f'sim. with {j}'] = dot_prod_compare_remake(j, train_image)
            df_test.loc[i, f'sim. with {j}']  = dot_prod_compare_remake(j, test_image)
        
        df_train.loc[i, 'label'] = i // 100
        df_test.loc[i, 'label']  = i // 100
    
    # change dataframes from float to int
    df_train['label'] = df_train['label'].astype(int)
    df_test['label']  = df_test['label'].astype(int)
    
    df_train.to_csv('feats_train.csv', index=False)
    df_test.to_csv('feats_test.csv'  , index=False)
    
if __name__ == "__main__":
    get_data_frames()
