# =============================================================================
# evaluates the performance of a random forest on the raw data, features data
# and the concatenation of the raw and feature data (raw | features)
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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

    return (y_pred != y_test).sum()

avg_raw  = 0
avg_feat = 0
avg_both = 0
n = 100
for i in range(n):
    model = RandomForestClassifier()
    avg_raw  += assess_pred(model, 'raw')           # best ~31
    avg_feat += assess_pred(model, 'feats')         # best ~26
    avg_both += assess_pred(model, 'raw_and_feats') # best ~22
print('avg_raw =' , avg_raw  / n)
print('avg_feat =', avg_feat / n)
print('avg_both =', avg_both / n)
