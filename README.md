# GradientBoostingRegressor
## _For genomic predictions_

>In this project, we aim to enhance the predictive performance of a [GradientBoostingRegression](https://github.com/scikit-learn/scikit-learn/) model for genomic predictions using the [EasyGeSe](https://github.com/stevenandrewyates/EasyGeSe/) database. EasyGeSe is a comprehensive genomic dataset that provides valuable insights into genetic variations and their associations with various phenotypic traits (Quesada _et al_., 2024).
## Parameters

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
When you are workin in an institute cluster or supercomputer  e.g. "euler" at the ETHZ, you need to be awared that you have dowloaded the last Python version and loaded.

```sh
module load python/3.7.4
python
```
## Example
This example can be done using the scrit bellow, to get 
