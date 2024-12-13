import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def init_weights(input_size, hidden_size, output_size):
    return {
        'hidden': np.zeros((input_size, hidden_size)),
        'output': np.zeros((hidden_size, output_size))
    }


def dynamic_learning_rate_schedule(initial_rate, decay, t):
    return initial_rate / (1 + (initial_rate / decay) * t)


def forward(X, weights):
    hidden_input = np.dot(X, weights['hidden'])
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights['output'])
    output = sigmoid(output_input)

    return hidden_output, output


def backward(X, y, output, hidden_output, weights, learning_rate):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights['output'].T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weights['output'] += np.outer(hidden_output, output_delta) * learning_rate
    weights['hidden'] += np.outer(X, hidden_delta) * learning_rate


def calculate_loss(X, y, weights):
    _, output = forward(X, weights)
    loss = np.mean((y - output) ** 2)
    return loss


def train_with_zero_weights(X_train, y_train, hidden_size, output_size, initial_lr, decay_rate, epochs):
    input_size = X_train.shape[1]
    weights = init_weights(input_size, hidden_size, output_size)
    loss_history = []

    for epoch in range(epochs):
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        for i in range(len(X_train)):
            X_instance = X_train[i]
            y_instance = y_train[i]

            learning_rate = dynamic_learning_rate_schedule(initial_lr, decay_rate, epoch * len(X_train) + i)

            hidden_output, output = forward(X_instance, weights)
            backward(X_instance, y_instance, output, hidden_output, weights, learning_rate)

            if i % 100 == 0:
                loss = calculate_loss(X_train, y_train, weights)
                loss_history.append(loss)

    return weights, loss_history


def test_model(X_test, y_test, weights):
    _, output = forward(X_test, weights)
    predictions = np.round(output)
    error = np.mean(predictions != y_test)
    return error


train_data = pd.read_csv("NeuralNetworks/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("NeuralNetworks/data/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

hidden_layer_sizes = [5, 10, 25, 50, 100]
initial_learning_rate = 0.5
learning_rate_decay = 0.002
epochs = 100

model_results = []

for hidden_size in hidden_layer_sizes:
    print(f"\nTraining with hidden layer size: {hidden_size} (Weights Initialized to Zero)")

    trained_weights, loss_curve = train_with_zero_weights(X_train, y_train, hidden_size, 1, initial_learning_rate, learning_rate_decay, epochs)

    train_error = test_model(X_train, y_train, trained_weights)

    test_error = test_model(X_test, y_test, trained_weights)

    model_results.append({
        'hidden_layer_size': hidden_size,
        'train_error': train_error,
        'test_error': test_error,
        'loss_curve': loss_curve
    })

    print(f"Training Error: {train_error * 100:.2f}% | Test Error: {test_error * 100:.2f}%")

plt.figure(figsize=(10, 6))
for result in model_results:
    plt.plot(result['loss_curve'], label=f"Hidden Layer Size: {result['hidden_layer_size']}")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve for Different Hidden Layer Sizes')
plt.legend()
plt.show()
