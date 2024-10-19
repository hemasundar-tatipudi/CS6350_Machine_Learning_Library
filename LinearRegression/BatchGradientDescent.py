import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('data/concrete/train.csv')
test_data = pd.read_csv('data/concrete/test.csv')

X_train = train_data.iloc[:, :-1].values 
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values


X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


def gradient_descent(X, y, w, learning_rate, tolerance, max_iterations=10000):
    cost_history = []
    iteration = 0
    while iteration < max_iterations:
        y_pred = np.dot(X, w)
        error = y_pred - y
        gradient = np.dot(X.T, error)
        w_new = w - learning_rate * gradient
        weight_change = np.linalg.norm(w_new - w)
        w = w_new
        cost = (1 / (2 * len(y))) * np.sum(error ** 2)
        cost_history.append(cost)
        if weight_change < tolerance:
            break
        iteration += 1
    return w, cost_history


w = np.zeros(X_train.shape[1])
learning_rate = 0.01
tolerance = 1e-6

final_w_batch, cost_history_batch = gradient_descent(X_train, y_train, w, learning_rate, tolerance)

plt.plot(range(1, len(cost_history_batch) + 1), cost_history_batch)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Batch Gradient Descent - Cost Function Convergence')
plt.show()

y_pred_test_batch = np.dot(X_test, final_w_batch)
test_cost_batch = (1 / (2 * len(y_test))) * np.sum((y_pred_test_batch - y_test) ** 2)
print(f"Batch Gradient Descent: Final weight vector: {final_w_batch}")
print(f"Test data cost function: {test_cost_batch}")
