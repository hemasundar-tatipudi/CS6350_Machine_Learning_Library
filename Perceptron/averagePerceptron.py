import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the datasets using Pandas
train_df = pd.read_csv('Perceptron/data/bank-note/train.csv', header=None)
test_df = pd.read_csv('Perceptron/data/bank-note/test.csv', header=None)

# Split the data into features (X) and labels (y)
X_train = train_df.iloc[:, :-1].values  # Training features
y_train = train_df.iloc[:, -1].values   # Training labels
X_test = test_df.iloc[:, :-1].values    # Test features
y_test = test_df.iloc[:, -1].values     # Test labels

# Feature scaling (standardize the data to mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize weight vectors for the current and average weights
num_features = X_train.shape[1]
weights = np.zeros(num_features)
avg_weights = np.zeros(num_features)

# Set the hyperparameters: learning rate and number of epochs
learning_rate = 0.1  # Increased learning rate
epochs = 50          # Increased number of epochs

# Train the Averaged Perceptron
for _ in range(epochs):
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train[i]
        
        # Update weights if the prediction is incorrect
        if yi * np.dot(weights, xi) <= 0:
            weights += learning_rate * yi * xi
        
        # Update the average weight vector
        avg_weights += weights

# Average the weight vector across all epochs
avg_weights /= (epochs * len(X_train))

# Calculate the average prediction error on the test set
error_count = 0
for i in range(len(X_test)):
    if y_test[i] * np.dot(avg_weights, X_test[i]) <= 0:
        error_count += 1

avg_error = error_count / len(X_test)

# Output the results
print("Learned Weight Vector (Average Weight Vector):", avg_weights)
print("Average Prediction Error on Test Data:", avg_error)
