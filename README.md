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
Let's look how the script works in Python:

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

```sh
SPECIES = 'lentil'#sys.argv[1]
TRAIT = 1#int(sys.argv[2])
CV = 1#int(sys.argv[3])
FOLD = 1#int(sys.argv[4])
```
Display input parameters.

```sh
print("Species: ", SPECIES)
print('Trait: ', TRAIT)
print('Cross validation: ', CV)
print('Fold: ', FOLD)
```
```sh
#Species:  lentil
#Trait:  1
#Cross validation:  1
#Fold:  1
```

Load data using an external script (LoadEasyGeSe.py):
```sh
with open("/~/GenoPredict/EasyGeSe/LoadEasyGeSe.py") as f:
    exec(f.read()) #Add the path of working directory
X, Y = LoadEasyGeSeData(SPECIES)
```
Loading 'lentil' database information:
```sh
Loaded 324 genotypes
Loaded 6 phenotyes
Loaded 23591 markers
Please cite:
Haile, Teketel A., et al. 'Genomic selection for lentil breeding: Empirical evidence.' The Plant Genome 13.1 (2020): e20002.
```
It is important to handel missing values in both arrays or DataFrames to avoid future errors:
```sh
Ydf = pd.DataFrame (Y)
Xdf = pd.DataFrame(X)
Ydf.replace("NA", np.nan, inplace=True)
Ydf = Ydf.dropna()
Xdf = Xdf.loc[Ydf.index]
Y = Ydf.to_numpy()
X = Xdf.to_numpy()
```
```sh
print(Y.shape)
#(324, 6)
print(X.shape)
#(324, 23591)
```
Create and write the output files (csv format) in the selected output directory.
```sh
output_directory = "/~/GenoPredict/GBR_results/"
filename = f"{output_directory}GenPred_estimators_{SPECIES}{TRAIT}{CV}{FOLD}.csv"
resfile = open(filename,"w")
csv_writer = csv.writer(resfile)
csv_writer.writerow(["SPECIES","TRAIT","CV", "FOLD", "cor", "rmse", "markers", "usefulmarkers", "estimators"])
all_results = [ ]
```
```sh
random.seed(CV)
print('setting up masks')
MASK = random.choices([1,2,3,4,5],k=X.shape[0])
print(MASK)
#[1, 5, 4, 2, 3, 3, 4, 4, 1, 1, 5, 3, 4, 1, 3, 4, 2, 5, 5, 1, 1, 3, 5, 2, 2, 3, 1, 2, 3, 3, 2, 2, 2, 3, 2, 1, 5, 3, 4, 1, 5, 5, 1, 2, 4, 4, 5, 3, 5, 4, 2, 3, 5, 5, 3, 3, 1, 2, 4, 3, 1, 3, 4, 4, 2, 3, 3, 4, 3, 2, 3, 1, 1, 4, 5, 3, 2, 1, 3, 5, 4, 3, 5, 2, 3, 5, 3, 3, 2, 3, 5, 1, 4, 5, 5, 4, 5, 3, 3, 3, 1, 5, 3, 1, 3, 3, 2, 2, 3, 4, 4, 3, 1, 2, 1, 3, 5, 4, 4, 5, 2, 5, 4, 1, 1, 1, 4, 2, 1, 4, 2, 1, 1, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 1, 5, 3, 2, 4, 5, 1, 1, 1, 4, 1, 4, 4, 3, 2, 5, 4, 3, 2, 4, 2, 3, 2, 4, 1, 2, 5, 5, 2, 5, 2, 5, 4, 3, 2, 1, 5, 1, 5, 5, 3, 1, 5, 5, 4, 3, 2, 2, 2, 4, 3, 1, 1, 4, 2, 3, 2, 5, 5, 1, 2, 2, 5, 4, 2, 2, 4, 5, 5, 2, 5, 4, 3, 5, 2, 4, 1, 1, 5, 2, 4, 4, 5, 2, 2, 2, 5, 4, 5, 5, 1, 3, 1, 1, 1, 5, 4, 5, 2, 4, 4, 2, 3, 2, 1, 2, 5, 3, 5, 3, 2, 4, 5, 1, 4, 1, 1, 5, 1, 2, 5, 3, 1, 1, 2, 4, 1, 5, 2, 5, 5, 2, 2, 3, 1, 4, 1, 1, 5, 2, 3, 3, 2, 1, 5, 5, 5, 1, 2, 4, 5, 3, 4, 4, 2, 3, 2, 2, 1, 2, 5, 3, 4, 4, 5, 2, 2, 2, 2, 5, 5, 2, 2, 3, 3, 3, 2, 1, 2, 1]
```
Selecting the train and test data sets to run the model base on the cross-validation values.
```sh
mm = np.array(MASK)
x_train = X[mm != FOLD,1:].astype('float64')
x_test = X[mm == FOLD,1:].astype('float64')
y_train = Y[mm != FOLD,TRAIT].astype('float64')
y_test = Y[mm == FOLD,TRAIT].astype('float64')
```
```sh
print(x_train.shape)
#(261, 23590)
print(x_test.shape)
#(63, 23590)
```
Let's run and fit the model!
```sh
reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
reg.fit(x_train,y_train)

#GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=3,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=1000,
                          n_iter_no_change=None, presort='auto', random_state=0,
                          subsample=1.0, tol=0.0001, validation_fraction=0.1,
                          verbose=0, warm_start=False)
```

Calculate the prediction values:
```sh
y_pred = reg.predict(x_test)
```
To evaluate the performance of our model, we calculate the coefficient correlation value and root mean squared error.

```sh
cor1 = np.corrcoef(y_test,y_pred)[0,1]
rmse1 = sqrt(mean_squared_error(y_test,y_pred))
print('correlation coefficient:', cor1)
#correlation coefficient: 0.7906774082910858
print('root squared error:', rmse1)
#root squared error: 2.1584768670835155
```
Give a feature importance of the markers used by the model to predict:
```sh
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
onlyimportantlogic = feature_importance > 0
onlyimportant = feature_importance[onlyimportantlogic]
p0 = sorted_idx[onlyimportantlogic]
usefulmarkers = p0.shape[0]
print(usefulmarkers)
#2273
```
