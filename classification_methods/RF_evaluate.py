# =============================================================================
# evaluates the performance of a random forest on the raw data, features data
# and the concatenation of the raw and feature data (raw | features) and
# performs the search for the best values for the parameters
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def assess_pred(model, data_form):
    
    # load data frames
    train_df = pd.read_csv(f"../{data_form}_train.csv")
    test_df = pd.read_csv(f"../{data_form}_test.csv")

    # split training data into features and label
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:,  -1]
    
    # split testing data into features and label
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:,  -1]

    # train/fit model
    model.fit(X_train, y_train)

    # use model to predict on test data
    y_pred = model.predict(X_test)

    wrong = (y_pred != y_test).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return wrong, accuracy

def splits_feats_search(n_trees = True, bstr = True, r_seed = None):
    '''
    WARNING: The full search takes about 15 minutes!

    Search for the best n_estimators, min_samples_split, and max_features
    and saves the results in a txt file.

    Depends on the way data is formatted above

    Defaults:
    n_trees = True means it search for ideal number of trees, otherwise uses 100 trees
    active bootstrap
    no random seed
    '''

    min_splits = [2, 3, 4, 5] # Viable values for min_samples_split
    max_fts = ['sqrt', 'log2'] # Viable values for max_features
    
    # Check for tree search
    if n_trees == True:
        trees = range(20,201) # Viable values for n_estimators
    else:
        trees = [100]

    datatype = ['raw', 'feats', 'raw_and_feats'] # Suffix to access the right data
    results = [] # Container for the results

    print('\n')
    print(f'Executing search for n_estimators = {trees}, min_splits = {min_splits} and max_features = {max_fts}...\n')

    for i in min_splits: # Cycle through min_splits
        for j in max_fts: # Cycle through max_features
            print(f'Working on (min_splits, max_features) = ({i}, {j}) ')
            
            # Results initialization for...
            draw = 0. # ... raw data
            dfeats = 0. # ... handcrafted features data
            drawnfeats = 0. # ... both raw and handcrafted features data

            for h in trees: # Cycle through n_estimators
                # Assess the model
                model = RandomForestClassifier(n_estimators=h, min_samples_split=i, max_features=j, bootstrap=True, random_state=r_seed)
                
                # Get the accuracy score for given model and data
                draw = assess_pred(model, datatype[0])[1]
                dfeats = assess_pred(model, datatype[1])[1]
                drawnfeats = assess_pred(model, datatype[2])[1]
                
                # Collect parameters' value and accuracies on data
                results.append([h, i, j, float(draw), float(dfeats), float(drawnfeats)])

    # Transform to array for easier use        
    results = np.array(results)

    # Results printing
    print('RESULTS:')
    print('tree | spl | mfts |  raw  | hc fs | r+hf |')
    print('------------------------------------------')
    for row in results:
        print('| {:^3} | {:^3} | {:^4} | {:4.5} | {:4.5} | {:4.5} |'.format(*row))
    
    # Save results in txt file
    np.savetxt('search_results.txt', results, fmt='%s', delimiter=',')
    return results

res_table = splits_feats_search(n_trees=True, r_seed=42)