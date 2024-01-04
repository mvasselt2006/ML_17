from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def classify(model, raw_or_feats):
    
    # load data frames
    train_df = pd.read_csv(f"../{raw_or_feats}_train.csv")
    test_df = pd.read_csv(f"../{raw_or_feats}_test.csv")

    # split training data into features and label
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    
    # split testing data into features and label
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # train/fit model
    model.fit(X_train, y_train)

    # use model to predict on test data
    y_pred = model.predict(X_test)
    
    # some conditional post-processing, needed for rounding in linear regression
    if isinstance(model, LinearRegression):
        y_pred = np.rint(model.predict(X_test))

    # number of inaccuracies
    print(f"\n\n{model} on {raw_or_feats} data -")
    print("total num. incorrect preds:", (y_pred != y_test).sum())
    for i in range(10):
        num_wrong_by_digit = (y_pred[100 * i : 100 * (i + 1)] != y_test[100 * i : 100 * (i + 1)]).sum()
        print(f"num. inccorect preds for {i}:", num_wrong_by_digit)
    
    # re-execute the function using the features data as well
    if raw_or_feats == 'raw':
        classify(model, 'feats')

model = LinearRegression()
classify(model, 'raw')

model = RandomForestClassifier()
classify(model, 'raw')

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
classify(model, 'raw')