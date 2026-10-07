# Polynomial Regression Assignment

## Overview

This project implements polynomial regression models for two engineering datasets.

Models evaluated:
- Ridge Regression
- Lasso Regression

Techniques used:
- Polynomial Feature Expansion
- Standard Scaling
- 5-Fold Cross Validation
- Hyperparameter tuning


## Dataset 1: var1

Polynomial degree range:
1-10

Best Model:
Lasso Regression

Alpha:
0.01

Degree:
5

CV MSE:
0.339408

CV R2:
0.963772


## Dataset 2: var2

Polynomial degree range:
1-20

Best Model:
Ridge Regression

Alpha:
1.0

Degree:
10

CV MSE:
0.240931

CV R2:
0.993902


## Running the Code

Install dependencies:

pip install -r requirements.txt


Run:

python src/model.py