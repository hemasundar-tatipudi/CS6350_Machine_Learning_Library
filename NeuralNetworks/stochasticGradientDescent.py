import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("NeuralNetworks/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("NeuralNetworks/data/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

widths = [5, 10, 25, 50, 100]
gamma_0 = 0.4
d = 0.001
epochs = 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(24)
    weights = {
        'hidden': np.random.randn(input_size, hidden_size) * 0.01,
        'output': np.random.randn(hidden_size, output_size) * 0.01
    }
    return weights

def learning_rate_schedule(gamma_0, d, t):
    return gamma_0 / (1 + (gamma_0 / d) * np.sqrt(t))

def forward_pass(X, weights):
    hidden_input = np.dot(X, weights['hidden'])
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights['output'])
    output = sigmoid(output_input)

    return hidden_output, output

def backward_pass(X, y, output, hidden_output, weights, learning_rate):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights['output'].T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weights['output'] += np.outer(hidden_output, output_delta) * learning_rate
    weights['hidden'] += np.outer(X, hidden_delta) * learning_rate

def calculate_loss(X, y, weights):
    _, output = forward_pass(X, weights)
    loss = np.mean((y - output) ** 2)
    return loss

def train_neural_network(X_train, y_train, hidden_size, output_size, gamma_0, d, epochs):
    input_size = X_train.shape[1]
    weights = initialize_weights(input_size, hidden_size, output_size)
    loss_curve = []

    for epoch in range(epochs):
        permutation = np.random.permutation(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        for i in range(len(X_train)):
            X = X_train[i]
            y = y_train[i]

            gamma_t = learning_rate_schedule(gamma_0, d, epoch * len(X_train) + i)
            
            hidden_output, output = forward_pass(X, weights)
            backward_pass(X, y, output, hidden_output, weights, gamma_t)

            if i % 100 == 0:
                loss = calculate_loss(X_train, y_train, weights)
                loss_curve.append(loss)

    return weights, loss_curve

def evaluate_neural_network(X_test, y_test, weights):
    _, output = forward_pass(X_test, weights)
    predictions = np.round(output)
    error = np.mean(predictions != y_test)
    return error

results = []

for hidden_size in widths:
    print(f"\nTraining with hidden layer size: {hidden_size}")

    trained_weights, loss_curve = train_neural_network(X_train, y_train, hidden_size, 1, gamma_0, d, epochs)

    train_error = evaluate_neural_network(X_train, y_train, trained_weights)
    test_error = evaluate_neural_network(X_test, y_test, trained_weights)

    results.append({
        'hidden_size': hidden_size,
        'train_error': train_error,
        'test_error': test_error,
        'loss_curve': loss_curve
    })

    print(f"Training Error: {train_error * 100:.2f}% | Test Error: {test_error * 100:.2f}%")


plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['loss_curve'], label=f"Hidden Size {result['hidden_size']}")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Learning Curves for Different Hidden Layer Sizes')
plt.legend()
plt.show()
