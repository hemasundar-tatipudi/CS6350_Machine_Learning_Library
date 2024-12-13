import numpy as np
import pandas as pd

train_data = pd.read_csv("LogisticRegression/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("LogisticRegression/data/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

num_epochs = 100
variance_values = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def dynamic_learning_rate(gamma_start, decay_rate, iteration):
    return gamma_start / (1 + (gamma_start / decay_rate) * iteration)

def negative_log_likelihood(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_gradient(y_true, y_pred, X):
    return -np.dot(X.T, (y_true - y_pred))


def fit_logistic_regression(X_train, y_train, gamma_start, decay_rate, epochs):
    num_features = X_train.shape[1]
    weights = np.zeros(num_features + 1)

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[shuffled_indices]
        y_train_shuffled = y_train[shuffled_indices]

        for i in range(len(X_train_shuffled)):
            feature_vector = np.concatenate([X_train_shuffled[i], [1]])
            label = y_train_shuffled[i]

            iteration = epoch * len(X_train_shuffled) + i
            learning_rate = dynamic_learning_rate(gamma_start, decay_rate, iteration)

            prediction = sigmoid_function(np.dot(weights, feature_vector))

            loss = negative_log_likelihood(label, prediction)

            gradient = compute_gradient(label, prediction, feature_vector)

            weights -= learning_rate * gradient

    return weights


def calculate_error_rate(weights, X, y):
    X_with_bias = np.column_stack([X, np.ones(len(X))])
    predictions = sigmoid_function(np.dot(X_with_bias, weights))
    predicted_classes = (predictions > 0.5).astype(int)
    error_rate = np.mean(predicted_classes != y)
    return error_rate


for variance in variance_values:
    print(f"\nTraining ML Estimation with Prior Variance: {variance}")
    
    final_weights_ml = fit_logistic_regression(X_train, y_train, gamma_start=0.1, decay_rate=0.01, epochs=num_epochs)

    training_error_ml = calculate_error_rate(final_weights_ml, X_train, y_train)
    print(f"Training Error (ML Estimation): {training_error_ml:.4f}")

    test_error_ml = calculate_error_rate(final_weights_ml, X_test, y_test)
    print(f"Test Error (ML Estimation): {test_error_ml:.4f}")
