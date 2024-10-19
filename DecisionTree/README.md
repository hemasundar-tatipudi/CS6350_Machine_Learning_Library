# Decision Tree Implementation

This folder contains an implementation of a Decision Tree algorithm for classification tasks.

## How to Use

1. **Import the Library**:
   ```python
   from decision_tree import DecisionTreeClassifier
2. **Initialize the Classifier**:
   ```python
    dtc = DecisionTreeClassifier(max_depth=5)
3. **Fit the Model**:
   ```python
    dtc.fit(X_train, y_train)
4. **Make Predictions**:
   ```python
    predictions = dtc.predict(X_test)

## Parameters
- max_depth: Maximum depth of the tree (default is None).
- min_samples_split: Minimum number of samples required to split an internal node (default is 2).