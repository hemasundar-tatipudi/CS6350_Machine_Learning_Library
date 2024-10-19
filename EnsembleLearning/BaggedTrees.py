import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

class TreeNode:
    def __init__(self, attribute, attribute_name, is_leaf, label, depth, info_gain, entropy_parent_attr, parent_attr_val):
        self.attribute = attribute
        self.attribute_name = attribute_name
        self.children = {}
        self.is_leaf = is_leaf
        self.label = label
        self.depth = depth
        self.info_gain = info_gain
        self.entropy_parent_attr = entropy_parent_attr
        self.parent_attr_val = parent_attr_val

    def predict(self, x):
        if self.is_leaf:
            return self.label
        current_val = x[self.attribute]
        if current_val not in self.children.keys():
            return self.label
        return self.children[current_val].predict(x)

class DecisionTreeClassifier:
    def __init__(self, max_depth=np.inf):
        self.root = None
        self.max_depth = max_depth

    def build_tree(self, X, Y, attribute_list, current_depth=0):
        if current_depth >= self.max_depth or len(attribute_list) == 0 or len(np.unique(Y)) == 1:
            vals, counts = np.unique(Y, return_counts=True)
            return TreeNode(None, None, True, vals[np.argmax(counts)], current_depth, None, None, None)

        max_info_gain = -1
        max_attribute = None

        for attribute in attribute_list:
            info_gain, _ = self.calculate_information_gain(X, Y, attribute)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_attribute = attribute

        vals, counts = np.unique(Y, return_counts=True)
        root = TreeNode(max_attribute, None, False, vals[np.argmax(counts)], current_depth, max_info_gain, None, None)

        attribute_values = np.unique(X[:, max_attribute])
        new_attribute_list = [attr for attr in attribute_list if attr != max_attribute]

        for value in attribute_values:
            indices = np.where(X[:, max_attribute] == value)[0]
            if len(indices) == 0:
                root.children[value] = TreeNode(None, None, True, vals[np.argmax(counts)], current_depth + 1, None, None, None)
            else:
                root.children[value] = self.build_tree(X[indices], Y[indices], new_attribute_list, current_depth + 1)

        return root

    def calculate_information_gain(self, X, Y, attribute):
        entropy_before_split = self.calculate_entropy(Y)
        values, counts = np.unique(X[:, attribute], return_counts=True)

        weighted_entropy_after_split = 0
        for i in range(len(values)):
            weighted_entropy_after_split += (counts[i] / np.sum(counts)) * self.calculate_entropy(Y[X[:, attribute] == values[i]])

        info_gain = entropy_before_split - weighted_entropy_after_split
        return info_gain, entropy_before_split

    def calculate_entropy(self, Y):
        _, counts = np.unique(Y, return_counts=True)
        probabilities = counts / len(Y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def fit(self, X, Y):
        attribute_list = list(range(X.shape[1]))
        self.root = self.build_tree(X, Y, attribute_list)

    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])

class BaggedTreesClassifier:
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.num_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]

            # Train a decision tree on the bootstrap sample
            dt_classifier = DecisionTreeClassifier(max_depth=np.inf)
            dt_classifier.fit(X_bootstrap, y_bootstrap)
            self.trees.append(dt_classifier)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.sign(np.sum(tree_predictions, axis=0))

# Load data
column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
dtype_dict = {'age': float, 'job': str, 'marital': str, 'education': str, 'default': str, 'balance': float, 'housing': str, 'loan': str, 'contact': str, 'day': float, 'month': str, 'duration': float, 'campaign': float, 'pdays': float, 'previous': float, 'poutcome': str, 'y': str }

train_file = "data/bank/train.csv"
test_file = "data/bank/test.csv"

train_df = pd.read_csv(train_file, names=column_headers, dtype=dtype_dict)
X_train = train_df.drop('y', axis=1).values
y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else -1).values

test_df = pd.read_csv(test_file, names=column_headers, dtype=dtype_dict)
X_test = test_df.drop('y', axis=1).values
y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else -1).values

# Vary the number of trees from 1 to 500
num_trees = np.arange(1, 501, 10)
train_errors = []
test_errors = []

for n in num_trees:
    bagged_trees = BaggedTreesClassifier(num_trees=n)
    bagged_trees.fit(X_train, y_train)
    y_train_pred = bagged_trees.predict(X_train)
    y_test_pred = bagged_trees.predict(X_test)
    
    train_errors.append(1 - accuracy_score(y_train, y_train_pred))
    test_errors.append(1 - accuracy_score(y_test, y_test_pred))

# Plot the errors
plt.plot(num_trees, train_errors, label='Training Error')
plt.plot(num_trees, test_errors, label='Test Error')
plt.title('Bagged Trees: Training vs Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.legend()
plt.grid()
plt.show()
