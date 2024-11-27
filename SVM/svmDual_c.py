import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("SVM/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("SVM/data/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

def gaussian_kernel(X, gamma):
    n = X.shape[0]
    kernel_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = X[i, :] - X[j, :]
            kernel_matrix[i, j] = np.exp(-np.dot(diff, diff) / (2 * gamma ** 2))
    return kernel_matrix

def train_dual_svm(X, y, C, gamma):
    K = gaussian_kernel(X, gamma)
    
    def objective(alpha):
        term1 = 0.5 * np.dot(alpha, np.dot(K, alpha * y) * y)
        term2 = -np.sum(alpha)
        return term1 + term2
    
    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y)}
    
    bounds = [(0, C)] * X.shape[0]
    
    initial_alpha = np.random.rand(X.shape[0]) * C  # Random initial values between 0 and C
    
    result = minimize(fun=objective, x0=initial_alpha, method='L-BFGS-B', bounds=bounds, constraints=constraints, options={'gtol': 1e-4})
    
    alphas = result.x
    
    support_vectors = alphas > 1e-5
    return support_vectors

C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]

print("Support Vectors Analysis:")
for C in C_values:
    for gamma in gamma_values:
        support_vectors = train_dual_svm(X_train, y_train, C, gamma)
        num_support_vectors = np.sum(support_vectors)
        print(f"C={C:.3f}, Gamma={gamma:.2f}: Support Vectors = {num_support_vectors}")
    print("-" * 60)

print("\nOverlapping Support Vectors (C=500/873):")
for i in range(len(gamma_values) - 1):
    gamma1 = gamma_values[i]
    gamma2 = gamma_values[i + 1]
    
    support_vectors_1 = train_dual_svm(X_train, y_train, 500 / 873, gamma1)
    support_vectors_2 = train_dual_svm(X_train, y_train, 500 / 873, gamma2)
    
    overlap_indices = np.where(np.logical_and(support_vectors_1, support_vectors_2))[0]
    num_overlap = len(overlap_indices)
    
    print(f"Gamma1={gamma1:.2f}, Gamma2={gamma2:.2f}: Overlapping Support Vectors = {num_overlap}")
