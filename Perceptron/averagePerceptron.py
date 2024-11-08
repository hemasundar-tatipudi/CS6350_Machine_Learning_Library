import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def averaged_perceptron(X_train, y_train, X_test, y_test, eta=0.1, num_epochs=50):
    current_weights = np.zeros(X_train.shape[1])
    cumulative_weights = np.zeros(X_train.shape[1])

    for epoch in range(num_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
        
        for features, label in zip(X_train_shuffled, y_train_shuffled):
            if label * np.dot(current_weights, features) <= 0:
                current_weights += eta * label * features
            cumulative_weights += current_weights

    avg_weight_vector = cumulative_weights / (num_epochs * len(X_train))

    misclassifications = sum(
        label * np.dot(avg_weight_vector, features) <= 0
        for features, label in zip(X_test, y_test)
    )
    test_error_rate = misclassifications / len(X_test)

    return avg_weight_vector, test_error_rate

train_data = pd.read_csv('Perceptron/data/bank-note/train.csv', header=None)
test_data = pd.read_csv('Perceptron/data/bank-note/test.csv', header=None)


X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

avg_weights, avg_test_error = averaged_perceptron(X_train, y_train, X_test, y_test)

print("Learned Weight Vector (Average Weight Vector):", avg_weights)
print(f"Average Prediction Error on Test Data: {avg_test_error:.3f}")
