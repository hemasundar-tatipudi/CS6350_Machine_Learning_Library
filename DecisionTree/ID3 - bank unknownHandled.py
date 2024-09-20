import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


column_headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
train_data = pd.read_csv("DecisionTree/data/bank/train.csv", names=column_headers)
test_data = pd.read_csv("DecisionTree/data/bank/test.csv", names=column_headers)


def fill_missing_with_majority(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            majority = df[col].mode()[0]  # Get the majority value
            df[col].replace('unknown', majority, inplace=True)  # Replace 'unknown' with majority
    return df

train_data = fill_missing_with_majority(train_data)
test_data = fill_missing_with_majority(test_data)

label_encoders = {}
for col in train_data.columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

for col in test_data.columns:
    if col in label_encoders:
        test_data[col] = test_data[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

X_train = X_train.astype(np.int64)
y_train = y_train.astype(np.int64)
X_test = X_test.astype(np.int64)
y_test = y_test.astype(np.int64)

class DecisionTreeNode:
    def __init__(self, is_leaf=False, prediction=None, attribute=None, branches=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.attribute = attribute
        self.branches = branches if branches is not None else {}

class DecisionTree:
    def __init__(self, criterion='information_gain', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def majority_error(self, y):
        counts = Counter(y)
        majority = counts.most_common(1)[0][1]
        return 1 - (majority / len(y))

    def gini_index(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum([p ** 2 for p in probabilities])

    def split_dataset(self, X, y, attribute):
        subsets = {}
        for val in np.unique(X[:, attribute]):
            idx = X[:, attribute] == val
            subsets[val] = (X[idx], y[idx])
        return subsets

    def calculate_gain(self, X, y, attribute):
        if self.criterion == 'information_gain':
            parent_entropy = self.entropy(y)
            weighted_entropy = sum(len(subset_y) / len(y) * self.entropy(subset_y)
                                   for subset_X, subset_y in self.split_dataset(X, y, attribute).values())
            return parent_entropy - weighted_entropy

        elif self.criterion == 'majority_error':
            parent_error = self.majority_error(y)
            weighted_error = sum(len(subset_y) / len(y) * self.majority_error(subset_y)
                                 for subset_X, subset_y in self.split_dataset(X, y, attribute).values())
            return parent_error - weighted_error

        elif self.criterion == 'gini':
            parent_gini = self.gini_index(y)
            weighted_gini = sum(len(subset_y) / len(y) * self.gini_index(subset_y)
                                for subset_X, subset_y in self.split_dataset(X, y, attribute).values())
            return parent_gini - weighted_gini

    def best_split(self, X, y):
        best_gain = -np.inf
        best_attribute = None
        for attribute in range(X.shape[1]):
            gain = self.calculate_gain(X, y, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
        return best_attribute

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return DecisionTreeNode(is_leaf=True, prediction=np.unique(y)[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(is_leaf=True, prediction=Counter(y).most_common(1)[0][0])

        best_attr = self.best_split(X, y)
        if best_attr is None:
            return DecisionTreeNode(is_leaf=True, prediction=Counter(y).most_common(1)[0][0])

        node = DecisionTreeNode(attribute=best_attr)
        for value, (subset_X, subset_y) in self.split_dataset(X, y, best_attr).items():
            node.branches[value] = self.build_tree(subset_X, subset_y, depth + 1)

        return node

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_single(self, x, node):
        if node.is_leaf:
            return node.prediction
        value = x[node.attribute]
        if value not in node.branches:
            return Counter(y_train).most_common(1)[0][0]
        return self.predict_single(x, node.branches[value])

    def predict(self, X):
        return [self.predict_single(x, self.root) for x in X]

def train_and_evaluate_tree(criterion, max_depth, X_train, y_train, X_test, y_test):
    model = DecisionTree(criterion=criterion, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    return train_error, test_error
    

criteria = ['information_gain', 'majority_error', 'gini']
max_depths = range(1, 17)
results = {criterion: [] for criterion in criteria}

for depth in max_depths:
    for criterion in criteria:
        train_error, test_error = train_and_evaluate_tree(criterion, depth, X_train, y_train, X_test, y_test)
        results[criterion].append((train_error, test_error))

print(f"{'Depth':<5} {'information_gain (train)':<25} {'information_gain (test)':<25} "
      f"{'majority_error (train)':<25} {'majority_error (test)':<25} {'gini (train)':<15} {'gini (test)':<15}")

for depth in max_depths:
    info_gain_train, info_gain_test = results['information_gain'][depth - 1]
    majority_error_train, majority_error_test = results['majority_error'][depth - 1]
    gini_train, gini_test = results['gini'][depth - 1]

    print(f"{depth:<5} {info_gain_train:<25.3f} {info_gain_test:<25.3f} "
          f"{majority_error_train:<25.3f} {majority_error_test:<25.3f} "
          f"{gini_train:<15.3f} {gini_test:<15.3f}")
