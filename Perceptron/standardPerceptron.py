import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def perceptron(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=10):
    np.random.seed(0)
    weights = np.random.normal(0, 0.01, X_train.shape[1])
    
    for _ in range(epochs):
        for i in range(len(X_train)):
            xi = X_train[i]
            yi = y_train[i]
            if yi * np.dot(weights, xi) <= 0:
                weights += learning_rate * yi * xi

    predictions = np.sign(np.dot(X_test, weights))
    
    errors = np.sum(predictions != y_test)
    avg_error = errors / len(X_test)
    
    return weights, avg_error


train_df = pd.read_csv('Perceptron/data/bank-note/train.csv', header=None)
test_df = pd.read_csv('Perceptron/data/bank-note/test.csv', header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

weights, avg_error = perceptron(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=10)

print("Learned Weight Vector:", weights)
print(f"Average Prediction Error on Test Data:, {avg_error:.3f}")
