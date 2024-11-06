import pandas as pd
import numpy as np

def train_voted_perceptron(X_train, y_train, X_test, y_test, eta=0.1, num_epochs=10, max_weights=10):
    """
    Trains a Voted Perceptron model and evaluates it on test data.
    
    Parameters:
    X_train (numpy.ndarray): Training dataset features.
    y_train (numpy.ndarray): Training dataset labels.
    X_test (numpy.ndarray): Test dataset features.
    y_test (numpy.ndarray): Test dataset labels.
    eta (float): Learning rate.
    num_epochs (int): Number of training epochs.
    max_weights (int): Max number of weight vectors to record.
    
    Returns:
    weight_vectors (list): List of up to `max_weights` weight vectors.
    count_correct (list): Count of correct predictions for each weight vector.
    avg_test_error (float): Average error rate on test data.
    """
    # Convert labels to {+1, -1} format for consistency
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Initialize weight vector and tracking variables
    weight = np.zeros(X_train.shape[1])
    weight_vectors = []
    count_correct = []

    # Training loop
    for epoch in range(num_epochs):
        correct_count = 1  # Initialize count of correct predictions for the current weight vector
        for xi, yi in zip(X_train, y_train):
            prediction = np.sign(np.dot(weight, xi))
            if prediction == 0:
                prediction = -1

            # Update weights on a mistake
            if prediction * yi <= 0:
                # Save weight vector only after first update and if max limit not reached
                if np.any(weight != 0) and len(weight_vectors) < max_weights:
                    weight_vectors.append(weight.copy())
                    count_correct.append(correct_count)
                
                # Update the weight vector
                weight += eta * yi * xi
                correct_count = 1  # Reset count after a misclassification
            else:
                correct_count += 1

        # Stop training once we've reached the max count of weight vectors
        if len(weight_vectors) == max_weights:
            break

    # Test phase: Calculate prediction errors using stored weight vectors
    test_error_rates = []
    for w, count in zip(weight_vectors, count_correct):
        errors = sum(yi != np.sign(np.dot(w, xi)) for xi, yi in zip(X_test, y_test))
        test_error_rates.append(errors / len(y_test))

    # Compute average test error
    avg_test_error = np.mean(test_error_rates)

    return weight_vectors, count_correct, avg_test_error

# Main block to load data, call the function, and display results
if __name__ == "__main__":
    # Load training and test data
    train_data = pd.read_csv('Perceptron/data/bank-note/train.csv', header=None)
    test_data = pd.read_csv('Perceptron/data/bank-note/test.csv', header=None)

    # Separate features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Run the voted perceptron model
    weight_vectors, count_correct, avg_test_error = train_voted_perceptron(X_train, y_train, X_test, y_test)

    # Display weight vectors and their correct prediction counts
    for i, (w, count) in enumerate(zip(weight_vectors, count_correct)):
        print(f"Weight Vector {i + 1}: {w}, Correct Count: {count}")

    # Show the average test error
    print(f"Average Test Error: {avg_test_error:.2f}")
