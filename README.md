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
When you are working in an Institute cluster or supercomputer  e.g. "euler" at the ETHZ, you need to be awared that you have dowloaded the last Python version and loaded it.

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
#                          learning_rate=0.1, loss='ls', max_depth=3,
#                          max_features=None, max_leaf_nodes=None,
#                          min_impurity_decrease=0.0, min_impurity_split=None,
#                          min_samples_leaf=1, min_samples_split=2,
#                          min_weight_fraction_leaf=0.0, n_estimators=1000,
#                          n_iter_no_change=None, presort='auto', random_state=0,
#                          subsample=1.0, tol=0.0001, validation_fraction=0.1,
#                          verbose=0, warm_start=False)
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
## Adding More Estimators? 
To improve the predictions values, we are going to evaluate the modification of hyperparameters such as number of estimator and the number of markers.
_As more estimators are added, the predictions of the models are combined to create a final and more accurate prediction._
> **NOTE:** You have to consider that this parameter can generate an overfitting model. 
```sh
result1= [SPECIES,TRAIT,CV, FOLD,cor1, rmse1, markers1, usefulmarkers, 1000]
all_results.append( result1 )
markers = [50,100, 200, 300, 400, 500, 1000, 2000]
```

Runing in a loop and saving results in a csv file.
```sh
for i in markers:
    new_markers = p0[-i:]
    new_markers = new_markers.astype(int)  # Convert new_markers to an array of integers
    x_train_new = x_train[:, new_markers].astype('float64')
    x_test_new = x_test[:, new_markers].astype('float64')
    reg_new = GradientBoostingRegressor(random_state=0, n_estimators=int(i/2))  # Convert i/2 to an integer
    reg_new.fit(x_train_new, y_train)
    y_pred_new = reg_new.predict(x_test_new)
    cor = np.corrcoef(y_test, y_pred_new)[0, 1]
    rmse = mean_squared_error(y_test, y_pred_new)
    feature_importance_new = reg_new.feature_importances_
    sorted_idx_new = np.argsort(feature_importance_new)
    onlyimportantlogic_new = feature_importance_new > 0
    onlyimportant_new = feature_importance_new[onlyimportantlogic_new]
    usefulmarkers_new = onlyimportant_new.shape[0]
    result = [SPECIES, TRAIT, CV, FOLD, cor, rmse, i, usefulmarkers_new, int(i/2)]  # Convert i/2 to an integer
    all_results.append(result)
csv_writer.writerows(all_results)
resfile.close()
```
## Results 
By now we got a csv file in GBR_results output directory in our terminal.
```sh
cd GBR_results
ls
#GenPred_estimators_lentil111.csv
cat GenPred_estimators_lentil111.csv
```
Inside **GenPred_estimators_lentil111.csv** file
```sh
SPECIES,TRAIT,CV,FOLD,cor,rmse,markers,usefulmarkers,estimators
lentil,1,1,1,0.7906774082910858,2.1584768670835155,23590,2273,1000
lentil,1,1,1,0.735192119502446,5.494681379013553,50,41,25
lentil,1,1,1,0.7946870737773924,4.3334386081124086,100,83,50
lentil,1,1,1,0.7891147899327088,4.645291985131125,200,160,100
lentil,1,1,1,0.7864034439829078,4.773130421945425,300,236,150
lentil,1,1,1,0.781889990290914,4.828922805362387,400,308,200
lentil,1,1,1,0.7787253174159432,4.723335496863951,500,372,250
lentil,1,1,1,0.779339497344807,4.801462582601394,1000,688,500
lentil,1,1,1,0.780212475892097,4.7550658984462775,2000,1302,1000
```
## R plot
>Plot using the plot_estimators.R in R

Plot of with the 20 cross-fold-validation for all traits of the 'lentil' dataset.
![](https://github.com/maeslopezortiz/GradientBoostingRegressor/blob/main/plots_lentil.png)
