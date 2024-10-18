import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define the DecisionTreeStump class from Code1
class DecisionTreeStump:
    def __init__(self, attribute, threshold, label_if_less, label_if_greater):
        self.attribute = attribute
        self.threshold = threshold
        self.label_if_less = label_if_less
        self.label_if_greater = label_if_greater

    def predict(self, x):
        if x[self.attribute] <= self.threshold:
            return self.label_if_less
        else:
            return self.label_if_greater

    @staticmethod
    def calculate_information_gain(X, y, attribute, threshold, weights):
        n = len(y)
        left_indices = X[:, attribute] <= threshold
        right_indices = X[:, attribute] > threshold

        left_weight = np.sum(weights[left_indices])
        right_weight = np.sum(weights[right_indices])

        if left_weight == 0 or right_weight == 0:
            return 0

        left_entropy = -np.sum(weights[left_indices] * np.log2(weights[left_indices] / left_weight))
        right_entropy = -np.sum(weights[right_indices] * np.log2(weights[right_indices] / right_weight))

        total_entropy = (left_weight / n) * left_entropy + (right_weight / n) * right_entropy
        return total_entropy

    @staticmethod
    def find_best_split(X, y, weights):
        num_features = X.shape[1]
        best_threshold = None
        best_attribute = None
        min_entropy = float('inf')

        for attribute in range(num_features):
            unique_values = np.unique(X[:, attribute])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                entropy = DecisionTreeStump.calculate_information_gain(X, y, attribute, threshold, weights)

                if entropy < min_entropy:
                    min_entropy = entropy
                    best_threshold = threshold
                    best_attribute = attribute

        return best_attribute, best_threshold


def train_adaboost(X, y, T=500):
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alphas = []
    
    train_errors = []
    test_errors = []
    stump_errors = []  # Store individual stump errors

    for t in range(T):
        # Train a decision stump (max_depth=1)
        attribute, threshold = DecisionTreeStump.find_best_split(X, y, weights)
        best_error = float('inf')
        best_stump = None

        for label_if_less in [-1, 1]:
            predictions = np.where(X[:, attribute] <= threshold, label_if_less, -label_if_less)
            error = np.sum(weights * (predictions != y))

            if error < best_error:
                best_error = error
                best_stump = DecisionTreeStump(attribute=attribute, threshold=threshold,
                                               label_if_less=label_if_less, label_if_greater=-label_if_less)

        classifiers.append(best_stump)

        # Calculate alpha (classifier weight)
        if best_error == 0:
            alpha = 1e-10
        else:
            alpha = 0.5 * np.log((1 - best_error) / max(best_error, 1e-10))
        
        alphas.append(alpha)

        # Update weights
        predictions = np.array([stump.predict(x) for stump in classifiers for x in X]).reshape(n_samples, -1)
        weights *= np.exp(-alpha * y * predictions[:, -1])  # using the last added stump's predictions
        weights /= np.sum(weights)  # Normalize the weights

        # Store weighted training error
        train_error = np.sum(weights * (predictions[:, -1] != y)) / np.sum(weights)
        train_errors.append(train_error)

        # Update test errors
        test_preds = adaboost_predict(X, classifiers, alphas)
        test_error = np.mean(test_preds != y)
        test_errors.append(test_error)

        # Store the stump error
        stump_errors.append(best_error)

    return classifiers, alphas, train_errors, test_errors, stump_errors

# Predict using the trained AdaBoost model
def adaboost_predict(X, classifiers, alphas):
    final_pred = np.zeros(X.shape[0])
    for clf, alpha in zip(classifiers, alphas):
        final_pred += alpha * np.array([clf.predict(x) for x in X])
    return np.sign(final_pred)

# Function to evaluate AdaBoost model
def evaluate_adaboost(X_train, y_train, X_test, y_test, T=500):
    classifiers, alphas, train_errors, test_errors, stump_errors = train_adaboost(X_train, y_train, T)
    
    train_preds = adaboost_predict(X_train, classifiers, alphas)
    test_preds = adaboost_predict(X_test, classifiers, alphas)
    
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    
    return train_accuracy, test_accuracy, train_errors, test_errors, stump_errors

# Function to generate plots for the training and test errors over iterations
def plot_errors(train_errors, test_errors, T):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, T + 1), train_errors, label='Training Error', color='blue')
    plt.plot(range(1, T + 1), test_errors, label='Test Error', color='orange')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error')
    plt.title('Training and Test Errors Over AdaBoost Iterations')
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot individual stump errors
def plot_individual_stump_errors(stump_errors):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(stump_errors) + 1), stump_errors, marker='o', label='Stump Error', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Training Errors of Individual Decision Stumps')
    plt.legend()
    plt.grid()
    plt.show()

# Main Function
def main():
    column_headers = ['age', 'job', 'marital', 'education', 'default', 
                      'balance', 'housing', 'loan', 'contact', 'day', 
                      'month', 'duration', 'campaign', 'pdays', 
                      'previous', 'poutcome', 'label']

    # Load your dataset with the specified column headers
    train_data = pd.read_csv("DecisionTree/data/bank/train.csv", names=column_headers, header=None)
    test_data = pd.read_csv("DecisionTree/data/bank/test.csv", names=column_headers, header=None)

    # Encode the labels as -1 and 1 (for AdaBoost)
    train_data['label'].replace({'no': -1, 'yes': 1}, inplace=True)
    test_data['label'].replace({'no': -1, 'yes': 1}, inplace=True)

    # Handle categorical features by one-hot encoding
    X_train = pd.get_dummies(train_data.drop('label', axis=1))
    y_train = train_data['label'].values
    X_test = pd.get_dummies(test_data.drop('label', axis=1))

    # Align the train and test data to have the same columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Train AdaBoost with decision stumps
    max_iters = 500
    train_accuracy, test_accuracy, train_errors, test_errors, stump_errors = evaluate_adaboost(X_train.values, y_train, X_test.values, test_data['label'].values, T=max_iters)

    print(f"Training accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    # Plot errors over the iterations
    plot_errors(train_errors, test_errors, max_iters)

    # Plot individual stump errors
    plot_individual_stump_errors(stump_errors)

if __name__ == "__main__":
    main()
