"""
Exploring a housing dataset and examining price distribution with supervised learning techniques.
"""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import grid_search


def load_data(filename):
    '''Load the dataset into numpy array.  
    We assume CSV with a header.'''

    data = np.genfromtxt(filename, delimiter=',')
    data = data[1:len(data),:]

    return data


def explore_data(data):
    '''Explore the housing statistics.'''

    # Assume target (home prices) are the last column of the dataset
    housing_prices = data[:,-1]

    # Assume all but last column of data are the features
    housing_features = data[:,0:data.shape[1]-1]

    # PRINT SOME HOUSING STATISTICS
    print "Exploring..."
    print "There are", housing_features.shape[0], "houses in your data."
    print "There are", housing_features.shape[1], "housing features in your data."
    print "The least-expensive home price in the data is: {0:.2f}" .format(housing_prices.min())
    print "The most-expensive home price in the data is: {0:.2f}" .format(housing_prices.max())
    print "The average home price in the data is: {0:.2f}" .format(np.mean(housing_prices))
    print "The mediam home price in the data is: {0:.2f}" .format(np.median(housing_prices))
    print "The standard deviation of a home price in the data is: {0:.2f}" .format(np.std(housing_prices))


def split_data(data, test_percentage):
    '''Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data.'''

    # Get the features and labels from the housing data
    X, y = data[:,0:data.shape[1]-1], data[:,-1]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size= test_percentage, random_state=0)

    return X_train, y_train, X_test, y_test


def mse(label, prediction):
    '''Calculate and return the MSE.'''

    return metrics.mean_squared_error(label,prediction)


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """
    Calculate the performance of the model after a set of training data.  
    
    - A learning curve is a visual graph that compares the metric performance
    of a model on training & testing data over a number of training instances.
    
    - When the testing curve and training curve plateau and there is no gap 
    the model has 'learned' everything 
    """

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth:", depth

    for i, s in enumerate(sizes):
        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = mse(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = mse(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
	'''Plot training and test error as a function of the training size.'''

	pl.figure()
	pl.title('Decision Trees: Performance vs Training Size')
	pl.plot(sizes, test_err, lw=2, label = 'test error')
	pl.plot(sizes, train_err, lw=2, label = 'training error')
	pl.legend()
	pl.xlabel('Training Size')
	pl.ylabel('Error')
	pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    '''
    Calculate the performance of the model as model complexity increases.

    - Model Complexity graph looks at how the complexity of a model changes
    the training and testing curves.

    - More Complexity -> More Variability
    '''

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
	    # Setup a Decision Tree Regressor so that it learns a tree with depth d
	    regressor = DecisionTreeRegressor(max_depth=d)

	    # Fit the learner to the training data
	    regressor.fit(X_train, y_train)

	    # Find the performance on the training set
	    train_err[i] = mse(y_train, regressor.predict(X_train))

	    # Find the performance on the testing set
	    test_err[i] = mse(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
	'''Plot training and test error as a function of the depth of the decision tree learn.'''

	pl.figure()
	pl.title('Decision Trees: Performance vs Max Depth')
	pl.plot(max_depth, test_err, lw=2, label = 'test error')
	pl.plot(max_depth, train_err, lw=2, label = 'training error')
	pl.legend()
	pl.xlabel('Max Depth')
	pl.ylabel('Error')
	pl.show()


def fit_predict_model(data, house):
    '''Find and tune the optimal model. Make a prediction on housing data.'''

    # Get the features and labels from the housing data
    X, y = data[:,0:data.shape[1]-1], data[:,-1]

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # define a performance metric
    mse = metrics.make_scorer(metrics.mean_squared_error,greater_is_better=False)

    # Use gridearch to fine tune the Decision Tree Regressor and find the best model
    reg = grid_search.GridSearchCV(regressor,parameters)
    reg.fit(X,y)

    # Fit the learner to the training data
    print "Final Model: "
    print reg.fit(X, y)
    
    # Use the model to predict the output of a particular sample
    x = house
    y = reg.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)


def main():
	'''Analyze the Boston housing data. Evaluate and validate the
	performanance of a Decision Tree regressor on the Boston data.
	Fine tune the model to make prediction on unseen data.'''

	# Load data
	city_data = load_data()

	# Explore the data
	explore_city_data(city_data)

	# Training/Test dataset split
	X_train, y_train, X_test, y_test = split_data(city_data)

	# Learning Curve Graphs
	max_depths = [1,2,3,4,5,6,7,8,9,10]
	for max_depth in max_depths:
		learning_curve(max_depth, X_train, y_train, X_test, y_test)

	# Model Complexity Graph
	model_complexity(X_train, y_train, X_test, y_test)

	# Tune and predict Model
	fit_predict_model(city_data)
