import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative_function(output):
    return output * (1 - output)

def initialize_weight_matrix(input_dim, hidden1_units, hidden2_units, output_dim):
    np.random.seed(0)
    return {
        'hidden1': np.random.randn(input_dim, hidden1_units),
        'hidden2': np.random.randn(hidden1_units, hidden2_units),
        'output': np.random.randn(hidden2_units, output_dim)
    }

def perform_forward_pass(inputs, weights):
    z1 = np.dot(inputs, weights['hidden1'])
    a1 = sigmoid_activation(z1)

    z2 = np.dot(a1, weights['hidden2'])
    a2 = sigmoid_activation(z2)

    z3 = np.dot(a2, weights['output'])
    output = sigmoid_activation(z3)

    return a1, a2, output

def compute_gradients(X, y, output, a2, a1, weights, learning_rate):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative_function(output)

    hidden2_error = output_delta.dot(weights['output'].T)
    hidden2_delta = hidden2_error * sigmoid_derivative_function(a2)

    hidden1_error = hidden2_delta.dot(weights['hidden2'].T)
    hidden1_delta = hidden1_error * sigmoid_derivative_function(a1)

    grad_output = np.outer(a2, output_delta)
    grad_hidden2 = np.outer(a1, hidden2_delta)
    grad_hidden1 = np.outer(X, hidden1_delta)

    weights['output'] += grad_output * learning_rate
    weights['hidden2'] += grad_hidden2 * learning_rate
    weights['hidden1'] += grad_hidden1 * learning_rate

def train_network(X_train, y_train, hidden1_units, hidden2_units, output_units, learning_rate, epochs):
    input_dim = X_train.shape[1]
    weights = initialize_weight_matrix(input_dim, hidden1_units, hidden2_units, output_units)

    for epoch in range(epochs):
        for i in range(X_train.shape[0]):
            X = X_train[i]
            y = y_train[i]

            a1, a2, output = perform_forward_pass(X, weights)
            compute_gradients(X, y, output, a2, a1, weights, learning_rate)

    return weights

def evaluate_model(X_test, y_test, weights):
    predictions = []
    for i in range(X_test.shape[0]):
        X = X_test[i]
        _, _, output = perform_forward_pass(X, weights)
        predictions.append(int(round(output[0])))

    accuracy = np.mean(predictions == y_test)
    return accuracy


train_data = pd.read_csv("NeuralNetworks/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("NeuralNetworks/data/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

hidden1_units = 4
hidden2_units = 4
output_units = 1
learning_rate = 0.001
epochs = 100

final_weights = train_network(X_train, y_train, hidden1_units, hidden2_units, output_units, learning_rate, epochs)

test_accuracy = evaluate_model(X_test, y_test, final_weights)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
