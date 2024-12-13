#!/bin/sh

echo "3a - MAP estimation"
python logistic_GradientDescent.py

echo "3b - maximum likelihood (ML) estimation"
python maxLikelihood.py