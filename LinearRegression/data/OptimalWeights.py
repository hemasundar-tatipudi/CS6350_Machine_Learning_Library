import numpy as np
import pandas as pd

train_data = pd.read_csv('LinearRegression/data/concrete/train.csv')
test_data = pd.read_csv('LinearRegression/data/concrete/test.csv')

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values


X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

optimal_w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_pred_test_optimal = np.dot(X_test, optimal_w)
test_cost_optimal = (1 / (2 * len(y_test))) * np.sum((y_pred_test_optimal - y_test) ** 2)

print(f"Analytical Solution: Optimal weight vector: {optimal_w}")
print(f"Test data cost function: {test_cost_optimal}")
