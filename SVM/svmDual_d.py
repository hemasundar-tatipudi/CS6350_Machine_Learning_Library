import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

train_data = pd.read_csv("SVM/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("SVM/data/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

X_train_data = train_data.iloc[:, :-1].values
y_train_data = train_data.iloc[:, -1].values
X_test_data = test_data.iloc[:, :-1].values
y_test_data = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train_data = scaler.fit_transform(X_train_data)
X_test_data = scaler.transform(X_test_data)

def gaussian_kernel_fn(x1, x2, gamma):
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

def kernel_perceptron_algorithm(X, y, gamma, iterations=100):
    num_samples = X.shape[0]
    alphas = np.zeros(num_samples)

    for _ in range(iterations):
        for i in range(num_samples):
            kernel_sum = np.sum(alphas * y * np.array([gaussian_kernel_fn(X[i], X[j], gamma) for j in range(num_samples)]))
            if y[i] * kernel_sum <= 0:
                alphas[i] += 1

    return alphas

def kernel_perceptron_predict_fn(X, X_train, y_train, alphas, gamma):
    num_samples = X.shape[0]
    predictions = np.zeros(num_samples)

    for i in range(num_samples):
        predictions[i] = np.sign(np.sum(alphas * y_train * np.array([gaussian_kernel_fn(X[i], X_train[j], gamma) for j in range(len(X_train))])))

    return predictions

def train_dual_svm_with_gaussian_kernel(X, y, C, sigma):
    num_samples = X.shape[0]
    kernel_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            kernel_matrix[i, j] = np.exp(-np.sum((X[i, :] - X[j, :]) ** 2) / (2 * sigma ** 2))

    def dual_objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(kernel_matrix, alpha * y) * y) - np.sum(alpha)

    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y), 'jac': lambda alpha: y}
    bounds = [(0, C) for _ in range(num_samples)]

    result = minimize(fun=dual_objective,
                      x0=np.zeros(num_samples),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    alphas = result.x
    support_vectors = (alphas > 1e-5)
    bias = np.mean(y[support_vectors] - np.dot(kernel_matrix[support_vectors], alphas * y))

    return alphas, bias, support_vectors

def svm_predict_with_kernel(X, X_support_vectors, y_support_vectors, alphas_support_vectors, bias, gamma):
    kernel_matrix = np.zeros((X.shape[0], X_support_vectors.shape[0]))

    for i in range(X.shape[0]):
        for j in range(X_support_vectors.shape[0]):
            kernel_matrix[i, j] = np.exp(-np.sum((X[i, :] - X_support_vectors[j, :]) ** 2) / (2 * gamma ** 2))

    predictions = np.dot(kernel_matrix, alphas_support_vectors * y_support_vectors) + bias
    return np.sign(predictions)

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100/873, 500/873, 700/873]

print("Kernel Perceptron Results:")
for gamma in gamma_values:
    alphas_kernel = kernel_perceptron_algorithm(X_train_data, y_train_data, gamma)
    train_pred_kernel = kernel_perceptron_predict_fn(X_train_data, X_train_data, y_train_data, alphas_kernel, gamma)
    test_pred_kernel = kernel_perceptron_predict_fn(X_test_data, X_train_data, y_train_data, alphas_kernel, gamma)

    train_error_kernel = np.mean(train_pred_kernel != y_train_data)
    test_error_kernel = np.mean(test_pred_kernel != y_test_data)

    print(f"Gamma: {gamma}, Train Error: {train_error_kernel:.5f}, Test Error: {test_error_kernel:.5f}")

print("\nNonlinear SVM with Gaussian Kernel Results:")
for C in C_values:
    for gamma in gamma_values:
        alphas_svm, bias_svm, support_vectors = train_dual_svm_with_gaussian_kernel(X_train_data, y_train_data, C, gamma)

        X_support_vectors = X_train_data[support_vectors]
        y_support_vectors = y_train_data[support_vectors]
        alphas_support_vectors = alphas_svm[support_vectors]

        train_pred_svm = svm_predict_with_kernel(X_train_data, X_support_vectors, y_support_vectors, alphas_support_vectors, bias_svm, gamma)
        test_pred_svm = svm_predict_with_kernel(X_test_data, X_support_vectors, y_support_vectors, alphas_support_vectors, bias_svm, gamma)

        train_error_svm = np.mean(train_pred_svm != y_train_data)
        test_error_svm = np.mean(test_pred_svm != y_test_data)

        print(f"Gamma: {gamma}, C: {C}, Train Error: {train_error_svm:.5f}, Test Error: {test_error_svm:.5f}")
