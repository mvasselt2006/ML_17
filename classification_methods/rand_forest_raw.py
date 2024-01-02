from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# load data frames
train_df = pd.read_csv("../raw_train.csv")
test_df = pd.read_csv("../raw_test.csv")

# split data into features and label
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# train/fit model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# use model to predict on test data
y_pred = rf_model.predict(X_test)

print("number of incorrect predictions:", (y_pred != y_test).sum())
for i in range(10):
    num_wrong_by_digit = (y_pred[100 * i : 100 * (i + 1)] != y_test[100 * i : 100 * (i + 1)]).sum()
    print(f"number of inccorect predictions for digit {i}:", num_wrong_by_digit)