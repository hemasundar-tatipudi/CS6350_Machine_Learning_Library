import pandas as pd
import numpy as np

def train_voted_perceptron(X_train, y_train, X_test, y_test, eta=0.1, num_epochs=10, max_weights=10):

    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    weight = np.zeros(X_train.shape[1])
    weight_vectors = []
    count_correct = []

    for epoch in range(num_epochs):
        correct_count = 1
        for xi, yi in zip(X_train, y_train):
            prediction = np.sign(np.dot(weight, xi))
            if prediction == 0:
                prediction = -1

            if prediction * yi <= 0:
                if np.any(weight != 0) and len(weight_vectors) < max_weights:
                    weight_vectors.append(weight.copy())
                    count_correct.append(correct_count)
                
                weight += eta * yi * xi
                correct_count = 1
            else:
                correct_count += 1

        if len(weight_vectors) == max_weights:
            break

    test_error_rates = []
    for w, count in zip(weight_vectors, count_correct):
        errors = sum(yi != np.sign(np.dot(w, xi)) for xi, yi in zip(X_test, y_test))
        test_error_rates.append(errors / len(y_test))

    avg_test_error = np.mean(test_error_rates)

    return weight_vectors, count_correct, avg_test_error


train_data = pd.read_csv('Perceptron/data/bank-note/train.csv', header=None)
test_data = pd.read_csv('Perceptron/data/bank-note/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

weight_vectors, count_correct, avg_test_error = train_voted_perceptron(X_train, y_train, X_test, y_test)

for i, (w, count) in enumerate(zip(weight_vectors, count_correct)):
    print(f"Weight Vector {i + 1}: {w}, Correct Count: {count}")

print(f"Average Test Error: {avg_test_error:.2f}")
