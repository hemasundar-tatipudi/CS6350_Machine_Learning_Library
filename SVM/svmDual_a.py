import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize

train_data = pd.read_csv("SVM/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("SVM/data/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].apply(lambda x: -1 if x == 0 else 1)
test_data.iloc[:, -1] = test_data.iloc[:, -1].apply(lambda x: -1 if x == 0 else 1)

def svm_sgd_new(X, y, C, initial_lr, decay_factor, epochs):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features) 
    bias = 0.0 
    iteration = 0

    for epoch in range(epochs):
        X, y = shuffle(X, y, random_state=epoch) 
        
        for j in range(num_samples):
            iteration += 1
            learning_rate = initial_lr / (1 + (iteration / decay_factor))
            decision_value = y[j] * (np.dot(X[j], weights) + bias)
            
            if decision_value < 1:
                weights = (1 - learning_rate) * weights + learning_rate * C * y[j] * X[j]
                bias += learning_rate * C * y[j]
            else:
                weights = (1 - learning_rate) * weights
    
    return weights, bias


def dual_loss_function(alpha, X, y, C):
    n_samples = len(X)
    weight_vector = np.dot(alpha * y, X)
    hinge_loss_term = np.maximum(1 - np.dot(X, weight_vector), 0)
    regularization_term = 0.5 * np.dot(weight_vector, weight_vector)
    
    return C * np.sum(hinge_loss_term) + regularization_term

alpha_initial = np.zeros(len(train_data))
regularization_values = [100/873, 500/873, 700/873]

primal_output = []
dual_output = []
for C in regularization_values:
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    w_primal, b_primal = svm_sgd_new(X_train, y_train, C, initial_lr=0.15, decay_factor=0.02, epochs=90)
    
    train_predictions_primal = np.sign(np.dot(X_train, w_primal) + b_primal)
    primal_train_error = np.mean(train_predictions_primal != y_train)
    
    test_predictions_primal = np.sign(np.dot(test_data.iloc[:, :-1].values, w_primal) + b_primal)
    primal_test_error = np.mean(test_predictions_primal != test_data.iloc[:, -1].values)
    
    primal_output.append({
        'C': C,
        'w': w_primal,
        'b': b_primal,
        'train_error': primal_train_error,
        'test_error': primal_test_error
    })
    
    result_dual = minimize(dual_loss_function, alpha_initial, args=(X_train, y_train, C), 
                           bounds=[(0, C) for _ in range(len(X_train))])
    
    alpha_opt = result_dual.x
    w_dual = np.dot(alpha_opt * y_train, X_train)
    b_dual_vals = y_train - np.dot(X_train, w_dual)
    b_dual = np.mean(b_dual_vals)
    
    train_predictions_dual = np.sign(np.dot(X_train, w_dual) + b_dual)
    dual_train_error = np.mean(train_predictions_dual != y_train)
    
    test_predictions_dual = np.sign(np.dot(test_data.iloc[:, :-1].values, w_dual) + b_dual)
    dual_test_error = np.mean(test_predictions_dual != test_data.iloc[:, -1].values)
    
    dual_output.append({
        'C': C,
        'w': w_dual,
        'b': b_dual,
        'train_error': dual_train_error,
        'test_error': dual_test_error
    })

for res_primal, res_dual in zip(primal_output, dual_output):
    C_value = res_primal['C']
    weight_diff = np.linalg.norm(res_primal['w'] - res_dual['w'])
    bias_diff = abs(res_primal['b'] - res_dual['b'])
    
    train_err_diff = abs(res_primal['train_error'] - res_dual['train_error'])
    test_err_diff = abs(res_primal['test_error'] - res_dual['test_error'])
    
    print(f"C: {C_value}")
    print("Primal SVM:")
    print("  w_primal:", res_primal['w'])
    print("  b_primal:", res_primal['b'])
    print("  Train Error: ", res_primal['train_error'])
    print("  Test Error: ", res_primal['test_error'])
    
    print("Dual SVM:")
    print("  w_dual:", res_dual['w'])
    print("  b_dual:", res_dual['b'])
    print("  Train Error: ", res_dual['train_error'])
    print("  Test Error: ", res_dual['test_error'])
    
    print("Differences:")
    print("  Weight Difference:", weight_diff)
    print("  Bias Difference:", bias_diff)
    print("  Train Error Difference:", train_err_diff)
    print("  Test Error Difference:", test_err_diff)
    print("-" * 40)
