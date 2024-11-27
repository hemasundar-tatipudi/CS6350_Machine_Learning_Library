import numpy as np
import pandas as pd
from random import shuffle

train_data = pd.read_csv("SVM/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("SVM/data/bank-note/test.csv", header=None)

train_data[4] = train_data[4].apply(lambda x: 1 if x == 1 else -1)
test_data[4] = test_data[4].apply(lambda x: 1 if x == 1 else -1)

num_epochs = 100 
C_values = [100 / 873, 500 / 873, 700 / 873]
initial_gamma = 0.1
param_a = 0.01 

def train_svm_sgd(features, labels, reg_param, init_gamma, a, epochs):
    num_samples, num_features = features.shape
    weight_vector = np.zeros(num_features)
    bias = 0 
    count = 0 
    
    for epoch in range(epochs):
        combined = list(zip(features, labels))
        shuffle(combined)
        features, labels = zip(*combined)
        
        for idx in range(num_samples):
            count += 1
            step_size = init_gamma / (1 + (init_gamma / a) * count)
            
            condition = labels[idx] * (np.dot(weight_vector, features[idx]) + bias)
            
            if condition < 1:
                weight_vector = (1 - step_size) * weight_vector + step_size * reg_param * labels[idx] * features[idx]
                bias += step_size * reg_param * labels[idx]
            else:
                weight_vector = (1 - step_size) * weight_vector

    return weight_vector, bias


def calculate_error(features, labels, weight_vector, bias):
    predictions = np.sign(np.dot(features, weight_vector) + bias)
    error = np.mean(predictions != labels)
    return error

for reg_param in C_values:
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    w_final, b_final = train_svm_sgd(X_train, y_train, reg_param, initial_gamma, param_a, num_epochs)
    
    train_err = calculate_error(X_train, y_train, w_final, b_final)
    test_err = calculate_error(X_test, y_test, w_final, b_final)
    
    print(f"Regularization Parameter (C): {reg_param}")
    print(f"Training Error: {train_err:.4f}, Test Error: {test_err:.4f}")
    print("-" * 40)
