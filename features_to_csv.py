# =============================================================================
# generates feats_train.csv and feats_test.csv
# =============================================================================

import numpy as np
import pandas as pd
import sys

sys.path.append('./handcrafted_features')

from hole_area import hole_area
from local_bin_pattern import lbp_features
from mix_of_gaussians import classify, train_individual
from proto_matching import dot_prod_compare_remake
from symmetry import hori_symmetry, vert_symmetry
from vertical_ratios import vert_ratio_image

df_raw_train = pd.read_csv('raw_train.csv')
df_raw_train = df_raw_train.iloc[:, :-1]

df_raw_test = pd.read_csv('raw_test.csv')
df_raw_test = df_raw_test.iloc[:, :-1]

def get_data_frames(train_data, test_data):
    
    df_train = pd.DataFrame()
    df_test  = pd.DataFrame()
    
    MOGs = train_individual(train_data)
    
    for i in range(1000):
        train_image = train_data[i]
        test_image  = test_data[i]
        
        # hole area feature
        df_train.loc[i, 'hole_area'] = hole_area(train_image)
        df_test.loc[i, 'hole_area']  = hole_area(test_image)
        
        # # MoG feature
        # df_train.loc[i, 'MoG'] = classify(train_image, MOGs)
        # df_test.loc[i, 'MoG']  = classify(test_image, MOGs)
        
        # prototype matching featue
        for j in range(10):
            df_train.loc[i, f'sim. with {j}'] = dot_prod_compare_remake(j, train_image, test_data)
            df_test.loc[i, f'sim. with {j}']  = dot_prod_compare_remake(j, test_image, test_data)
        
        # symmetry (horizontal) feature
        df_train.loc[i, 'hor_sym'] = hori_symmetry(train_image)
        df_test.loc[i, 'hor_sym'] = hori_symmetry(test_image)
        
        # symmetry (vertical) feature
        df_train.loc[i, 'vert_sym'] = vert_symmetry(train_image)
        df_test.loc[i, 'vert_sym'] = vert_symmetry(test_image)
        
        # vertical ratio feature
        df_train.loc[i, 'vert_ratio'] = vert_ratio_image(train_image, 8)
        df_test.loc[i, 'vert_ratio'] = vert_ratio_image(test_image, 8)
        
        # local binary pattern
        lbp_train = lbp_features(train_image)
        lbp_test = lbp_features(test_image)
        for j in range(10):
            df_train.loc[i, f'LBP_{j}'] = lbp_train[j]
            df_test.loc[i, f'LBP_{j}'] = lbp_test[j]
        
        # labels
        df_train.loc[i, 'label'] = i // 100
        df_test.loc[i, 'label']  = i // 100
    
    # change label column from float to int
    df_train['label'] = df_train['label'].astype(int)
    df_test['label']  = df_test['label'].astype(int)
    
    # save features dataframe to csv
    df_train.to_csv('feats_train.csv', index=False)
    df_test.to_csv('feats_test.csv'  , index=False)
    
    # save concatentation of raw and feature dataframes to csv
    pd.concat([df_raw_train, df_train], axis=1).to_csv('raw_and_feat_train.csv', index=False)
    pd.concat([df_raw_test, df_test], axis=1).to_csv('raw_and_feat_test.csv', index=False)
    
if __name__ == "__main__":
    
    # load data
    with open('data.txt', 'r') as file:
        dataset = np.array([list(map(int, line.split())) for line in file])

    # parition data into training and testing datasets
    train_data = np.empty((0, 240))
    test_data  = np.empty((0, 240))
    for i in range(10):
        train_data = np.vstack((train_data, dataset[200 * i : 200 * i + 100]))
        test_data  = np.vstack((test_data , dataset[200 * i + 100 : 200 * i + 200]))
    
    get_data_frames(train_data, test_data)
