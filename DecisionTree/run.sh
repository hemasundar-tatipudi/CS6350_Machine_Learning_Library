#!/bin/sh

echo "Evaluation metrics for train and test in Car dataset"
python3 ID3 - car evaluation.py

echo "Evaluation metrics for train and test when unknown is considered as a feature value in Bank dataset"
python3 ID3 - bank unknown.py

echo "Evaluation metrics for train and test when unknown is considered as a missing value in Bank dataset"
python3 ID3 - bank unknownHandled.py