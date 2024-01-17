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

    # # number of inaccuracies
    # print(f"\n\n{model} on {data_form} data -")
    # print("total num. incorrect preds:", (y_pred != y_test).sum())
    # for i in range(10):
    #     num_wrong_by_digit = (y_pred[100 * i : 100 * (i + 1)] != y_test[100 * i : 100 * (i + 1)]).sum()
    #     print(f"num. inccorect preds for {i}:", num_wrong_by_digit)

    return (y_pred != y_test).sum()

avg_raw  = 0
avg_feat = 0
avg_both = 0
n = 1
for i in range(n):
    model = RandomForestClassifier()
    avg_raw  += assess_pred(model, 'raw')           # best ~31 (close to constant as raw data is constant)
    avg_feat += assess_pred(model, 'feats')         # best ~25
    avg_both += assess_pred(model, 'raw_and_feats') # best ~21
print('avg_raw =' , avg_raw  / n)
print('avg_feat =', avg_feat / n)
print('avg_both =', avg_both / n)
