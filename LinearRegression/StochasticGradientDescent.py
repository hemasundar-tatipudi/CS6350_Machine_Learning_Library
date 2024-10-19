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


def stochastic_gradient_descent(X, y, w, learning_rate, max_iterations):
    cost_history = []
    for _ in range(max_iterations):
        random_index = np.random.randint(0, len(y))
        xi = X[random_index]
        yi = y[random_index]
        y_pred = np.dot(xi, w)
        error = y_pred - yi
        gradient = xi * error
        w -= learning_rate * gradient
        y_pred_all = np.dot(X, w)
        cost = (1 / (2 * len(y))) * np.sum((y_pred_all - y) ** 2)
        cost_history.append(cost)
    return w, cost_history

w = np.zeros(X_train.shape[1])
learning_rate = 0.001
max_iterations = 10000

final_w_sgd, cost_history_sgd = stochastic_gradient_descent(X_train, y_train, w, learning_rate, max_iterations)

plt.plot(range(1, len(cost_history_sgd) + 1), cost_history_sgd)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Stochastic Gradient Descent - Cost Function Convergence')
plt.show()

y_pred_test_sgd = np.dot(X_test, final_w_sgd)
test_cost_sgd = (1 / (2 * len(y_test))) * np.sum((y_pred_test_sgd - y_test) ** 2)
print(f"Stochastic Gradient Descent: Final weight vector: {final_w_sgd}")
print(f"Test data cost function: {test_cost_sgd}")
