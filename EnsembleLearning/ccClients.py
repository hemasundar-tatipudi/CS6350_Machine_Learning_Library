import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
data = pd.read_excel(url, header=1, index_col=0)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=24000, test_size=6000, random_state=42)

n_iterations = 500
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n_iterations, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=n_iterations, random_state=42)
#adaboost_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_iterations, random_state=42)
adaboost_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_iterations, algorithm='SAMME', random_state=42)
tree_clf = DecisionTreeClassifier()

train_errors_bagging = []
test_errors_bagging = []
train_errors_rf = []
test_errors_rf = []
train_errors_adaboost = []
test_errors_adaboost = []
train_errors_tree = []
test_errors_tree = []

for i in range(1, n_iterations + 1):
    bagging_clf.n_estimators = i
    bagging_clf.fit(X_train, y_train)
    train_errors_bagging.append(1 - accuracy_score(y_train, bagging_clf.predict(X_train)))
    test_errors_bagging.append(1 - accuracy_score(y_test, bagging_clf.predict(X_test)))
    
    rf_clf.n_estimators = i
    rf_clf.fit(X_train, y_train)
    train_errors_rf.append(1 - accuracy_score(y_train, rf_clf.predict(X_train)))
    test_errors_rf.append(1 - accuracy_score(y_test, rf_clf.predict(X_test)))
    
    adaboost_clf.n_estimators = i
    adaboost_clf.fit(X_train, y_train)
    train_errors_adaboost.append(1 - accuracy_score(y_train, adaboost_clf.predict(X_train)))
    test_errors_adaboost.append(1 - accuracy_score(y_test, adaboost_clf.predict(X_test)))

tree_clf.fit(X_train, y_train)
train_errors_tree.append(1 - accuracy_score(y_train, tree_clf.predict(X_train)))
test_errors_tree.append(1 - accuracy_score(y_test, tree_clf.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iterations + 1), train_errors_bagging, label='Bagging (Train)', linestyle='--', color='blue')
plt.plot(range(1, n_iterations + 1), test_errors_bagging, label='Bagging (Test)', color='blue')
plt.plot(range(1, n_iterations + 1), train_errors_rf, label='Random Forest (Train)', linestyle='--', color='green')
plt.plot(range(1, n_iterations + 1), test_errors_rf, label='Random Forest (Test)', color='green')
plt.plot(range(1, n_iterations + 1), train_errors_adaboost, label='Adaboost (Train)', linestyle='--', color='red')
plt.plot(range(1, n_iterations + 1), test_errors_adaboost, label='Adaboost (Test)', color='red')
plt.axhline(y=train_errors_tree[0], color='orange', linestyle='--', label='Single Tree (Train)')
plt.axhline(y=test_errors_tree[0], color='orange', label='Single Tree (Test)')

plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Training and Test Errors for Bagged Trees, Random Forest, and Adaboost')
plt.legend()
plt.grid(True)
plt.show()
