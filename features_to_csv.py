# =============================================================================
# generates feats_{train/test}.csv and raw_and_feats_{train/test}.csv
# =============================================================================

import numpy  as np
import pandas as pd
import sys
import warnings

# just to silence a specific warning that clogs the output console
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# needed to import the inclded handcrafted features while in this directory
sys.path.append('./handcrafted_features')

from gabor_filter    import gabor_features
from HoGs            import HoG_descriptor
from hole_area       import hole_area
from proto_matching  import dot_prod_compare_remake
from row_col_count   import col_count_nonzero, row_count_nonzero
from row_col_sum     import col_count_sum, row_count_sum
from vertical_ratios import vert_ratio_image

# extract train data
df_raw_train = pd.read_csv('raw_train.csv')
df_raw_train = df_raw_train.iloc[:, :-1]

# extract test data
df_raw_test = pd.read_csv('raw_test.csv')
df_raw_test = df_raw_test.iloc[:, :-1]

# recompute all included features and store in feats_{train/test}.csv 
# and raw_and_feats_{train/test}.csv
def get_data_frames(train_data, test_data):
    
    # initialise train and test datafames
    df_train = pd.DataFrame()
    df_test  = pd.DataFrame()
    
    for i in range(1000):
        
        # current iterations train/test image
        train_image = train_data[i]
        test_image  = test_data[i]
        
        # include hole area feature
        df_train.loc[i, 'hole_area'] = hole_area(train_image)
        df_test.loc[i, 'hole_area']  = hole_area(test_image)
        
        # include vertical ratio feature with k = 3
        df_train.loc[i, 'vert_ratio_3'] = vert_ratio_image(train_image, 3)
        df_test.loc[i, 'vert_ratio_3']  = vert_ratio_image(test_image, 3)
        
        # include vertical ratio feature with k = 8
        df_train.loc[i, 'vert_ratio_8'] = vert_ratio_image(train_image, 8)
        df_test.loc[i, 'vert_ratio_8']  = vert_ratio_image(test_image, 8)
        
        # include column count feature
        for j in range(15):
            df_train.loc[i, f'col_count_{j}'] = col_count_nonzero(train_image, j)
            df_test.loc[i, f'col_count_{j}']  = col_count_nonzero(test_image, j)
            
        # include row count feature
        for j in range(16):
            df_train.loc[i, f'row_count_{j}'] = row_count_nonzero(train_image, j)
            df_test.loc[i, f'row_count_{j}']  = row_count_nonzero(test_image, j)
        
        # include column sum feature
        for j in range(15):
            df_train.loc[i, f'col_sum_{j}'] = col_count_sum(train_image, j)
            df_test.loc[i, f'col_sum_{j}']  = col_count_sum(test_image, j)
        
        # include row sum feature
        for j in range(16):
            df_train.loc[i, f'row_sum_{j}'] = row_count_sum(train_image, j)
            df_test.loc[i, f'row_sum_{j}']  = row_count_sum(test_image, j)
        
        # include gabor filter feature
        gabor_train = gabor_features(train_image)
        gabor_test  = gabor_features(test_image)
        for j in range(8):
            df_train.loc[i, f'gabor_{j}'] = gabor_train[j]
            df_test.loc[i, f'gabor_{j}']  = gabor_test[j]
            
        # include hog descriptor feature
        hog_descriptor_train = HoG_descriptor(train_image)
        hog_descriptor_test  = HoG_descriptor(test_image)
        for j in range(36):
            df_train.loc[i, f'hog_descriptor_{j}'] = hog_descriptor_train[j]
            df_test.loc[i, f'hog_descriptor_{j}']  = hog_descriptor_test[j]
        
        # include labels
        df_train.loc[i, 'label'] = i // 100
        df_test.loc[i, 'label']  = i // 100
    
    # change label column from floats to ints
    df_train['label'] = df_train['label'].astype(int)
    df_test['label']  = df_test['label'].astype(int)
    
    # load pototype data frame
    df_proto_train = pd.read_csv('proto_train.csv')
    df_proto_test  = pd.read_csv('proto_test.csv')
    
    # attach prototype data frame to the left of feats data frame
    df_train = pd.concat([df_proto_train, df_train], axis=1)
    df_test  = pd.concat([df_proto_test, df_test], axis=1)
    
    # save features dataframe to csv
    df_train.to_csv('feats_train.csv', index=False)
    df_test.to_csv('feats_test.csv'  , index=False)
    
    # save concatentation of raw and feature dataframes to csv
    pd.concat([df_raw_train, df_train], axis=1).to_csv('raw_and_feats_train.csv', index=False)
    pd.concat([df_raw_test, df_test], axis=1).to_csv('raw_and_feats_test.csv', index=False)

# this is the most time consuming feature to re-compute each time so instead
# compute it once, store it then concatenate to complete feats_{train/test}.csv
def get_proto_df(train_data, test_data):
    
    # initialise train and test datafames
    df_train = pd.DataFrame()
    df_test  = pd.DataFrame()
    
    for i in range(1000):
        
        # current iterations train/test image
        train_image = train_data[i]
        test_image  = test_data[i]
        
        # include prototype matching featue
        for j in range(10):
            df_train.loc[i, f'sim. with {j}'] = dot_prod_compare_remake(j, train_image, test_data)
            df_test.loc[i, f'sim. with {j}']  = dot_prod_compare_remake(j, test_image, test_data)
        
    # save features dataframe to csv
    df_train.to_csv('proto_train.csv', index=False)
    df_test.to_csv('proto_test.csv'  , index=False)
    
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
    
    # get_proto_df(train_data, test_data)
    get_data_frames(train_data, test_data)
