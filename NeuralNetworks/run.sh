#!/bin/sh

echo "2a - Back Propagation"
python backPropagation.py

echo "2b - Gradient Descent"
python stochasticGradientDescent.py

echo "2c - Gradient Descent with zero weights"
python stochasticGradientDescent_modifiedW.py

echo "2e - Neural Network using PyTorch"
python NN_pytorch.py