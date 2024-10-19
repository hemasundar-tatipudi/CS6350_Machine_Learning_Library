#!/bin/sh

echo "Bagged Tree"
python3 DecisionTree/BaggedTrees.py

echo "Results for Bias and Variance"
python3 DecisionTree/BiasVarianceDecomposition.py

echo "Results for Bias and Variance (including RandomForest)"
python3 DecisionTree/BiasVarianceDecomposition2.py

echo "Adaboost"
python3 DecisionTree/DecisionStumps_AdaBoost.py

echo "Randomforest"
python3 DecisionTree/randomforest.py

echo "ccClients DataSet"
python3 DecisionTree/ccClients.py

echo "Batch Gradient Descent"
python3 LinearRegression/BatchGradientDescent.py

echo "Stochastic Gradient Descent"
python3 LinearRegression/StochasticGradientDescent.py

echo "Optimal Weights"
python3 LinearRegression/OptimalWeights.py

echo "Evaluation metrics for train and test in Car dataset"
python3 DecisionTree/ID3 - car evaluation.py

echo "Evaluation metrics for train and test when unknown is considered as a feature value in Bank dataset"
python3 DecisionTree/ID3 - bank unknown.py

echo "Evaluation metrics for train and test when unknown is considered as a missing value in Bank dataset"
python3 DecisionTree/ID3 - bank unknownHandled.py