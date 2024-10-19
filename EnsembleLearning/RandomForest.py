import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
column_headers = ['age', 'job', 'marital', 'education', 'default', 
                  'balance', 'housing', 'loan', 'contact', 'day', 
                  'month', 'duration', 'campaign', 'pdays', 
                  'previous', 'poutcome', 'label']
dtype_dict = {
    'age': float,
    'job': str,
    'marital': str,
    'education': str,
    'default': str,
    'balance': float,
    'housing': str,
    'loan': str,
    'contact': str,
    'day': float,
    'month': str,
    'duration': float,
    'campaign': float,
    'pdays': float,
    'previous': float,
    'poutcome': str,
    'label': str
}
train_file = "data/bank/train.csv"
test_file = "data/bank/test.csv"
train_df = pd.read_csv(train_file, names=column_headers, dtype=dtype_dict)
X_train = train_df.drop('label', axis=1)
y_train = train_df['label'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

test_df = pd.read_csv(test_file, names=column_headers, dtype=dtype_dict)
X_test = test_df.drop('label', axis=1)
y_test = test_df['label'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)


label_encoders = {}
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column].astype(str))
    X_test[column] = le.transform(X_test[column].astype(str))
    label_encoders[column] = le


class RandomForestClassifier:
    def __init__(self, num_trees, max_features, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        for _ in range(self.num_trees):
            selected_features = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X[:, selected_features]

            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap, Y_bootstrap = X_subset[indices], Y[indices]

            dt_classifier = DecisionTreeClassifier(max_depth=10)
            dt_classifier.fit(X_bootstrap, Y_bootstrap)
            self.trees.append((dt_classifier, selected_features))

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for dt_classifier, selected_features in self.trees:
            X_subset = X[:, selected_features]
            predictions += dt_classifier.predict(X_subset)
        return np.sign(predictions)


num_trees_range = range(1, 501)
max_features_range = [2, 4, 6]

train_errors_rf = {2: [], 4: [], 6: []}
test_errors_rf = {2: [], 4: [], 6: []}

for max_features in max_features_range:
    for num_trees in num_trees_range:
        rf_classifier = RandomForestClassifier(num_trees, max_features)
        rf_classifier.fit(X_train.values, y_train)

        y_train_pred = rf_classifier.predict(X_train.values)
        y_test_pred = rf_classifier.predict(X_test.values)

        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        train_errors_rf[max_features].append(train_error)
        test_errors_rf[max_features].append(test_error)


plt.figure(figsize=(12, 6))
for max_features in max_features_range:
    plt.plot(num_trees_range, train_errors_rf[max_features], label=f'Train Error (max_features={max_features})')
    plt.plot(num_trees_range, test_errors_rf[max_features], label=f'Test Error (max_features={max_features})', linestyle='dashed')


plt.xlabel('Number of Random Trees')
plt.ylabel('Error')
plt.title('Random Forest: Train and Test errors vs Number of Trees')
plt.legend()
plt.grid()
plt.show()
