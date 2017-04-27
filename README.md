# Predicting Home Prices Using Supervised Learning

>*NOTE: Sample is included in this repository.

The Jupyter notebook `homeprice_prediction.ipynb` contains a walkthrough of an accompanying Python module (`homeprice_prediction.py`) for predicting home prices using supervised learning through [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree).  It contains code for displaying ['learning curves'](https://en.wikipedia.org/wiki/Learning_curve#In_machine_learning) so that a user can view how the model's predictions perform on unseen homes as more and more training data is analyzed.

![alt text](http://i.imgur.com/wojpXgx.jpg)

The notebook also contains code for displaying ['model complexity graphs'](http://www.dummies.com/programming/big-data/data-science/model-complexity-machine-learning/) to guard against overfitting.

![alt text](http://i.imgur.com/7ZfcXds.jpg)

The notebook then concludes with code that optimizes the Decision Tree over a range of parameters and demonstrates a prediction of the optimized model with a sample house.

## Software and Library Requirements
* Python 2.7.11
* Jupyter Notebook 4.2.2
* Numpy 1.11.2
* scikit-image 0.12.3
* matplotlib 1.5.2

## Data

The sample data used for this notebook comes from the [UCI Irvine Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/Housing).

## Getting up and running

While in the `homeprice_prediction` directory, use the following command in your command line interface:

> `ipython notebook homeprice_prediction.ipynb`
