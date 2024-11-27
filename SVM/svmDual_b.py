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
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def gaussian_kernel_matrix(X, gamma):
    """Compute the RBF (Gaussian) kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-gamma * np.sum((X[i, :] - X[j, :]) ** 2))
    return K

def train_dual_svm_gaussian(X, y, C, gamma):
    """Train an SVM in the dual form using the Gaussian kernel."""
    n_samples = X.shape[0]
    
    K = gaussian_kernel_matrix(X, gamma)
    
    def dual_objective(alpha):
        return 0.5 * np.sum((alpha * y) @ K @ (alpha * y)) - np.sum(alpha)
    
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y), 'jac': lambda alpha: y}
    bounds = [(0, C) for _ in range(n_samples)]
    
    result = minimize(fun=dual_objective,
                      x0=np.zeros(n_samples),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'ftol': 1e-10, 'disp': False})
    
    alphas = result.x
    sv = alphas > 1e-5
    
    b = np.mean(y[sv] - np.sum((alphas[sv] * y[sv])[:, None] * K[sv][:, sv], axis=1))
    
    return alphas, b, sv


def svm_predict(X, X_sv, y_sv, alphas_sv, b, gamma):
    """Predict using the trained SVM with support vectors and RBF kernel."""
    n_samples = X.shape[0]
    n_sv = X_sv.shape[0]
    K = np.zeros((n_samples, n_sv))
    
    for i in range(n_samples):
        for j in range(n_sv):
            K[i, j] = np.exp(-gamma * np.sum((X[i, :] - X_sv[j, :]) ** 2))
    
    predictions = np.dot(K, alphas_sv * y_sv) + b
    return np.sign(predictions)

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100/873, 500/873, 700/873]

best_error = float('inf')
best_params = None

for gamma in gamma_values:
    for C in C_values:
        alphas, b, sv = train_dual_svm_gaussian(X_train, y_train, C, gamma)
        
        X_sv = X_train[sv]
        y_sv = y_train[sv]
        alphas_sv = alphas[sv]
        
        y_train_pred = svm_predict(X_train, X_sv, y_sv, alphas_sv, b, gamma)
        y_test_pred = svm_predict(X_test, X_sv, y_sv, alphas_sv, b, gamma)
        
        train_error = np.mean(y_train_pred != y_train)
        test_error = np.mean(y_test_pred != y_test)
        
        print(f"For C={C:.5f} and gamma={gamma:.5f}:")
        print(f"  Training error: {train_error:.5f}")
        print(f"  Test error: {test_error:.5f}")
        print("-" * 40)
        
        if test_error < best_error:
            best_error = test_error
            best_params = {'gamma': gamma, 'C': C}

print(f"\nBest Parameters: {best_params}, Best Test Error: {best_error:.5f}")
