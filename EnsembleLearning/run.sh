#!/bin/sh

echo "Bagged Tree"
python3 BaggedTrees.py

echo "Results for Bias and Variance"
python3 BiasVarianceDecomposition.py

echo "Results for Bias and Variance (including RandomForest)"
python3 BiasVarianceDecomposition2.py

echo "Adaboost"
python3 DecisionStumps_AdaBoost.py

echo "Randomforest"
python3 randomforest.py


echo "ccClients DataSet"
python3 ccClients.py