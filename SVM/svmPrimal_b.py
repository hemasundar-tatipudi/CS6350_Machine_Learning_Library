import numpy as np
import pandas as pd
from sklearn.utils import shuffle

train_data = pd.read_csv("SVM/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("SVM/data/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)
test_data.iloc[:, -1] = test_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)

max_iter = 100
C_values = [100 / 873, 500 / 873, 700 / 873]
initial_gamma = 0.1

def svm_gradient_descent(data, labels, C, initial_gamma, max_iter):
    num_samples, num_features = data.shape
    weights = np.zeros(num_features)
    bias = 0
    step_count = 0

    for iteration in range(max_iter):
        data, labels = shuffle(data, labels, random_state=iteration)
        
        for index in range(num_samples):
            step_count += 1
            current_eta = initial_gamma / (1 + step_count)
            
            product = labels[index] * (np.dot(data[index], weights) + bias)
            
            if product < 1:
                weights = (1 - current_eta) * weights + current_eta * C * labels[index] * data[index]
                bias += current_eta * C * labels[index]
            else:
                weights = (1 - current_eta) * weights

    return weights, bias

for C in C_values:
    train_X, train_y = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    test_X, test_y = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    final_weights, final_bias = svm_gradient_descent(train_X, train_y, C, initial_gamma, max_iter)
    
    train_predictions = np.sign(np.dot(train_X, final_weights) + final_bias)
    train_error_rate = np.mean(train_predictions != train_y)
    
    test_predictions = np.sign(np.dot(test_X, final_weights) + final_bias)
    test_error_rate = np.mean(test_predictions != test_y)
    
    print(f"Regularization Parameter(C): {C}")
    print(f"Training Error: {train_error_rate:.4f}, Test Error: {test_error_rate:.4f}")
    print("-" * 30)
