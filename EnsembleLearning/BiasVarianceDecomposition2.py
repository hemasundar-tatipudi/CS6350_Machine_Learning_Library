import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


columns_bank_dataset = [
    ("age", "numeric"),
    ("job", "categorical"),
    ("marital", "categorical"),
    ("education", "categorical"),
    ("default", "categorical"),
    ("balance", "numeric"),
    ("housing", "categorical"),
    ("loan", "categorical"),
    ("contact", "categorical"),
    ("day", "numeric"),
    ("month", "categorical"),
    ("duration", "numeric"),
    ("campaign", "numeric"),
    ("pdays", "numeric"),
    ("previous", "numeric"),
    ("poutcome", "categorical"),
    ("label", "categorical"),
]

target_variable = "label"

def load_and_preprocess_data(file, columns_structure, target_variable):
    dtype_dict = {}
    column_headers = [col[0] for col in columns_structure]

    for col, col_type in columns_structure:
        dtype_dict[col] = float if col_type == "numeric" else str 

    df = pd.read_csv(file, names=column_headers, dtype=dtype_dict)
    df[target_variable] = df[target_variable].apply(lambda x: 1 if x == 'yes' else -1)
    
    X = df.drop(target_variable, axis=1).values
    y = df[target_variable].values

    return X, y


class NodeTree:
    def __init__(self, attribute, attribute_name, is_leaf, value, depth, info_gain):
        self.attribute = attribute
        self.attribute_name = attribute_name
        self.is_leaf = is_leaf
        self.value = value
        self.depth = depth
        self.info_gain = info_gain
        self.children = {}

    def add_child(self, child_node, value):
        self.children[value] = child_node

    def predict(self, x):
        if self.is_leaf:
            return self.value
        value = x[self.attribute]
        if value in self.children:
            return self.children[value].predict(x)
        else:
            return self.value


class DecisionTree:
    def __init__(self, max_depth=np.inf):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, Y):  
        attribute_names = list(range(X.shape[1])) 
        attribute_list = np.arange(X.shape[1])
        self.root = self.build_tree(X, Y, attribute_names, attribute_list, 0)

    def build_tree(self, X, Y, attribute_names, attribute_list=[], current_depth=0):
        if current_depth >= self.max_depth or len(attribute_list) == 0 or len(np.unique(Y)) == 1:
            vals, counts = np.unique(Y, return_counts=True)
            return NodeTree(None, None, True, vals[np.argmax(counts)], current_depth, None)

        max_info_gain = -1
        max_attribute = None
        for attribute in attribute_list:
            info_gain, _, _ = self.info_gain(X, Y, attribute)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_attribute = attribute

        vals, counts = np.unique(Y, return_counts=True)
        root = NodeTree(max_attribute, attribute_names[max_attribute],
                        False, vals[np.argmax(counts)], current_depth, max_info_gain)

        attribute_values = np.unique(X[:, max_attribute])
        new_attribute_list = np.delete(attribute_list, np.where(attribute_list == max_attribute))
        for value in attribute_values:
            indices = np.where(X[:, max_attribute] == value)[0]
            if len(indices) == 0:
                root.add_child(NodeTree(None, None, True, vals[np.argmax(counts)], current_depth + 1, max_info_gain), value)
            else:
                root.add_child(self.build_tree(X[indices], Y[indices], attribute_names, new_attribute_list, current_depth + 1), value)
        return root

    def info_gain(self, X, Y, attribute):
        _, counts = np.unique(Y, return_counts=True)
        entropy_attribute = self.entropy_calc(counts)
        entropy_parent = 0
        distinct_attr_values = np.unique(X[:, attribute])
        for val in distinct_attr_values:
            indices = np.where(X[:, attribute] == val)[0]
            _, counts = np.unique(Y[indices], return_counts=True)
            entr = self.entropy_calc(counts)
            entropy_parent += (len(indices) / len(Y)) * entr
        return entropy_attribute - entropy_parent, entropy_attribute, entropy_parent

    def entropy_calc(self, counts):
        total = sum(counts)
        return -sum((element / total) * np.log2(element / total) for element in counts if element != 0)

    def predict(self, X):
        return np.array([self.root.predict(X[i]) for i in range(X.shape[0])])


class BaggedTrees:
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.num_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]

            dt_classifier = DecisionTree(max_depth=np.inf)
            dt_classifier.fit(X_bootstrap, y_bootstrap) 
            self.trees.append(dt_classifier)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += tree.predict(X)
        return np.sign(predictions)


class RandomForest:
    def __init__(self, num_trees, max_features):
        self.num_trees = num_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.num_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_bootstrap, y_bootstrap = X[indices][:, feature_indices], y[indices]

            tree = DecisionTree(max_depth=np.inf)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree, feature_indices in self.trees:
            predictions += tree.predict(X[:, feature_indices])
        return np.sign(predictions)


def bias_variance(predictions, ground_truth):
    """Calculate bias and variance for the predictions."""
    bias = np.mean(predictions) - ground_truth
    variance = np.var(predictions)
    return bias, variance

def squared_error(predictions, ground_truth):
    """Calculate squared error."""
    return np.mean((predictions - ground_truth) ** 2)

def estimate_bias_variance(X_train, y_train, X_test, y_test, num_trees, max_features):
    bagged_classifier = BaggedTrees(num_trees)
    bagged_classifier.fit(X_train, y_train)
    bagged_predictions = bagged_classifier.predict(X_test)

    bagged_bias, bagged_variance = bias_variance(bagged_predictions, y_test)
    bagged_se = squared_error(bagged_predictions, y_test)

    single_tree_predictions = bagged_classifier.trees[0].predict(X_test)

    single_tree_bias, single_tree_variance = bias_variance(single_tree_predictions, y_test)
    single_tree_se = squared_error(single_tree_predictions, y_test)

    rf_classifier = RandomForest(num_trees, max_features)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)

    rf_bias, rf_variance = bias_variance(rf_predictions, y_test)
    rf_se = squared_error(rf_predictions, y_test)

    return (np.mean(single_tree_bias), single_tree_variance, single_tree_se), \
           (np.mean(bagged_bias), bagged_variance, bagged_se), \
           (np.mean(rf_bias), rf_variance, rf_se)


train_file = "data/bank/train.csv"
test_file = "data/bank/test.csv"

X_train, y_train = load_and_preprocess_data(train_file, columns_bank_dataset, target_variable)
X_test, y_test = load_and_preprocess_data(test_file, columns_bank_dataset, target_variable)


num_trees = 100
max_features = 4  # You can vary this

(single_tree_bias, single_tree_variance, single_tree_se), \
(bagged_bias, bagged_variance, bagged_se), \
(rf_bias, rf_variance, rf_se) = estimate_bias_variance(X_train, y_train, X_test, y_test, num_trees, max_features)

print(f"Single Random Tree -> Bias: {single_tree_bias}, Variance: {single_tree_variance}, Squared Error: {single_tree_se}")
print(f"Bagged Tree -> Bias: {bagged_bias}, Variance: {bagged_variance}, Squared Error: {bagged_se}")
print(f"Random Forest -> Bias: {rf_bias}, Variance: {rf_variance}, Squared Error: {rf_se}")
