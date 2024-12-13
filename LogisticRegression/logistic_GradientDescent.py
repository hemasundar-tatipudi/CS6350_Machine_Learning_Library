import numpy as np
import pandas as pd

train_data = pd.read_csv("LogisticRegression/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("LogisticRegression/data/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

epochs = 100
prior_variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]


def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

def adaptive_learning_rate(lr_0, decay_rate, iteration):
    return lr_0 / (1 + (lr_0 / decay_rate) * iteration)

def compute_loss(y_true, y_pred, weights, variance):
    log_likelihood = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    regularization_term = (1 / (2 * variance)) * np.sum(weights**2)  # Gaussian regularization
    return log_likelihood + regularization_term

def compute_gradient(y_true, y_pred, features, weights, variance):
    grad = -np.dot(features.T, (y_true - y_pred)) + (1 / variance) * weights
    return grad


def logistic_regression_with_prior(X_data, y_data, variance, initial_lr, decay_rate, max_epochs):
    num_features = X_data.shape[1]
    weights = np.zeros(num_features + 1)

    for epoch in range(max_epochs):
        shuffled_indices = np.random.permutation(len(X_data))
        X_data_shuffled = X_data[shuffled_indices]
        y_data_shuffled = y_data[shuffled_indices]

        for i in range(len(X_data_shuffled)):
            X_instance = np.concatenate([X_data_shuffled[i], [1]])
            y_instance = y_data_shuffled[i]

            lr_at_t = adaptive_learning_rate(initial_lr, decay_rate, epoch * len(X_data_shuffled) + i)

            predictions = sigmoid_activation(np.dot(weights, X_instance))

            loss = compute_loss(y_instance, predictions, weights, variance)

            grad = compute_gradient(y_instance, predictions, X_instance, weights, variance)
            weights -= lr_at_t * grad

    return weights


def calculate_error(weights, X_data, y_data):
    X_with_bias = np.column_stack([X_data, np.ones(len(X_data))]) 
    predictions = sigmoid_activation(np.dot(X_with_bias, weights))
    predicted_labels = (predictions > 0.5).astype(int)
    error_rate = np.mean(predicted_labels != y_data)
    return error_rate


for variance in prior_variances:
    print(f"\nTraining with Variance: {variance}")
    final_weights = logistic_regression_with_prior(X_train, y_train, variance, initial_lr=0.1, decay_rate=0.01, max_epochs=epochs)

    training_error = calculate_error(final_weights, X_train, y_train)
    print(f"Training Error: {training_error:.4f}")

    test_error = calculate_error(final_weights, X_test, y_test)
    print(f"Test Error: {test_error:.4f}")
