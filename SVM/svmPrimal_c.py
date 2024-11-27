import numpy as np
import pandas as pd
from sklearn.utils import shuffle

train_data = pd.read_csv("SVM/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("SVM/data/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].replace({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].replace({1: 1, 0: -1})

num_epochs = 100
C_vals = [100 / 873, 500 / 873, 700 / 873]
gamma_init_a = 0.1
a_param = 0.01
gamma_init_t = 0.1

def svm_training(data_X, data_y, reg_param, gamma_init, a_param, epochs, sched_type):
    n_samples, n_features = data_X.shape
    weight_vec = np.zeros(n_features)
    bias_term = 0
    step_count = 0

    for epoch in range(epochs):
        data_X, data_y = shuffle(data_X, data_y, random_state=epoch)
        
        for j in range(n_samples):
            step_count += 1
            
            if sched_type == 'a':
                learning_rate = gamma_init / (1 + (gamma_init / a_param) * step_count)
            elif sched_type == 't':
                learning_rate = gamma_init / (1 + step_count)
            else:
                raise ValueError("Invalid schedule type provided!")
            
            decision_margin = data_y[j] * (np.dot(weight_vec, data_X[j]) + bias_term)
            
            if decision_margin < 1:
                weight_vec = (1 - learning_rate) * weight_vec + learning_rate * reg_param * data_y[j] * data_X[j]
                bias_term += learning_rate * reg_param * data_y[j]
            else:
                weight_vec = (1 - learning_rate) * weight_vec
    
    return weight_vec, bias_term

results_schedule_a = []
for C in C_vals:
    X_train, y_train = train_data.iloc[:, :-1].to_numpy(), train_data.iloc[:, -1].to_numpy()
    X_test, y_test = test_data.iloc[:, :-1].to_numpy(), test_data.iloc[:, -1].to_numpy()
    
    w_final, b_final = svm_training(X_train, y_train, C, gamma_init_a, a_param, num_epochs, 'a')
    
    train_pred_a = np.sign(np.dot(X_train, w_final) + b_final)
    train_err_a = np.mean(train_pred_a != y_train)
    
    test_pred_a = np.sign(np.dot(X_test, w_final) + b_final)
    test_err_a = np.mean(test_pred_a != y_test)
    
    results_schedule_a.append({
        'C': C,
        'weights': w_final,
        'bias': b_final,
        'train_err': train_err_a,
        'test_err': test_err_a
    })

results_schedule_t = []
for C in C_vals:
    X_train, y_train = train_data.iloc[:, :-1].to_numpy(), train_data.iloc[:, -1].to_numpy()
    X_test, y_test = test_data.iloc[:, :-1].to_numpy(), test_data.iloc[:, -1].to_numpy()
    
    w_final, b_final = svm_training(X_train, y_train, C, gamma_init_t, a_param, num_epochs, 't')
    
    train_pred_t = np.sign(np.dot(X_train, w_final) + b_final)
    train_err_t = np.mean(train_pred_t != y_train)
    
    test_pred_t = np.sign(np.dot(X_test, w_final) + b_final)
    test_err_t = np.mean(test_pred_t != y_test)
    
    results_schedule_t.append({
        'C': C,
        'weights': w_final,
        'bias': b_final,
        'train_err': train_err_t,
        'test_err': test_err_t
    })

for result_a, result_t in zip(results_schedule_a, results_schedule_t):
    print(f"Regularization Parameter (C): {result_a['C']}")
    
    training_diff = abs(result_a['train_err'] - result_t['train_err'])
    testing_diff = abs(result_a['test_err'] - result_t['test_err'])
    print(f"  Training Error Difference: {training_diff}, Test Error Difference: {testing_diff}")
    
    weight_diff = np.linalg.norm(result_a['weights'] - result_t['weights'])
    bias_diff = abs(result_a['bias'] - result_t['bias'])
    print(f"  Weight Difference: {weight_diff}, Bias Difference: {bias_diff}")
    
    print("-" * 30)
