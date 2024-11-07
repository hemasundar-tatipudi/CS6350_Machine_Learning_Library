import pandas as pd
import numpy as np

def perceptron(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=10):
    
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    weights = np.zeros(X_train.shape[1])

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for x, y in zip(X_train_shuffled, y_train_shuffled):
            if y * np.dot(weights, x) <= 0:
                weights += learning_rate * y * x

    test_errors = sum(y_test[i] * np.dot(weights, X_test[i]) <= 0 for i in range(len(X_test)))
    average_error = test_errors / len(X_test)

    return weights, average_error



train_data = pd.read_csv('Perceptron/data/bank-note/train.csv', header=None)
test_data = pd.read_csv('Perceptron/data/bank-note/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

weights, average_error = perceptron(X_train, y_train, X_test, y_test)

print("Learned Weight Vector:", weights)
print("Average Prediction Error on Test Data:", average_error)
