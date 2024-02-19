# GradientBoostingRegressor
## _For genomic predictions_

>In this project, we aim to enhance the predictive performance of a [GradientBoostingRegression](https://github.com/scikit-learn/scikit-learn/) model for genomic predictions using the [EasyGeSe](https://github.com/stevenandrewyates/EasyGeSe/) database. EasyGeSe is a comprehensive genomic dataset that provides valuable insights into genetic variations and their associations with various phenotypic traits (Quesada _et al_., 2024).
## Features

Selecting the parameters which are applied to increase the prediction accuracy.

| Parameter | values |
| ------ | ------ |
| Fold: | 5 |
| Cross-Validation: | 20 (Repeating the process). |
| Average the scores: | mean sequared error (MSE) and correlation coefficient (cor). |
| Testing no. estimators: | all, 2000, 1000, 500, 400, 300, 200, 100, 50. |
| Testing no. markers: | 1000, 500, 250, 200, 150, 100, 50, 25. |


## Download _EasyGeSe_
Download the EasyGeSe folder in your working directory:
```sh
git clone https://github.com/stevenandrewyates/EasyGeSe
mkdir GBR_results #create output directoty for results
```
## Working in Python
When you are working in an Institute cluster or supercomputer  e.g. "euler" at the ETHZ, you need to be awared that you have dowloaded the last Python version and loaded.

```sh
module load python/3.7.4
python
```
## Example
This example can be done using the scriot bellow, to get coefficient correlation values and mean squared error from the data base 'lentil'.
You can run the Python script as:
```sh
#python script.py species_name trait_column cross-fold-validation
python  GBR_na.py 'lentil' 1 1 1 
```
Let's look how the script works:

```sh
#!/usr/bin/env python3
from sklearn.ensemble import GradientBoostingRegressor  # Import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # Import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for array operations
import csv  # Import CSV library for file handling
import random  # Import random for random sampling
import sys
import pandas as pd
from math import sqrt
```
