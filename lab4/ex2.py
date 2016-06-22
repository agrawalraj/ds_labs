
#Author: Raj Agrawal 

"""
Lab 4: Cross-Validation 

In this lab we will implement different functions to do cross-validation
from scratch without using sklearn. We will do cross-validation in the 
setting of linear-regression but this workflow is the nearly same for any 
type of machine learning algorithm. Cross-validation is a great way to not
only pick hyperparameters for a given model (e.g. like the amount of regularization 
in Ridge Regression) but also as a model selection tool. 

We will use sklearn's linear models library to fit the linear models and
work with the diabetes dataset.  

References:
See notes. 
"""

import numpy as np

from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression 
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes['data']
Y = diabetes['target']

def MSE(prediction, actual):
	"""
	input:
	    prediction - n x 1 np array
	    actual - n x 1 np array 
	
	return: 
	    mse - the MSE between the predicted and actual values in a np array.  
	"""
	#YOUR CODE HERE

def kFoldCV(Y, X, k, model, random_state=42):
	"""
	"""
	#Randomly shuffle indces for split  
	num_obs = len(Y)
	indcs = np.arange(0, num_obs)
	np.random.seed(random_state)
    np.random.shuffle(indcs)
    
    #YOUR CODE HERE

    #Divide shuffled array into equal sized groups 
    size_of_each_group = num_obs / k 
    groups = []
    for i in range(0, k):
    	group_i_indcs = indcs[(size_of_each_group*i):((i+1)*size_of_each_group)]
        groups.append(group_i_indcs)
    
    #Choose group i as testing set and the rest as training.  
    #Record errors and repeat for i = 1,...,k   
    errors = []
    indcs_set = set(indcs)
    for i, validation_set_indcs in groups:
    	training_set = indcs_set.difference(set(validation_set_indcs)) 
    	fit_model = model(Y[training, X[training, ], lam)
    	predictions = fit_model(Y[validation_set_indcs])
    	actual = Y[validation_set_indcs]
    	error = mse(predictions, actual)
    	errors.append(error)
    return mean(errors)
   
################# Do Cross-Validation #################

#Cross Validation to choose good lambda values
SET_SEED = 42
np.random.seed(SET_SEED)
lambdas = range(.1, 10, .1)
#lambda_ridge = CV param. chosen w/ k = 14
#lambda_lasso = CV param. chosen w/ k = 14
#mse_ridge = 
#mse_lasso = 
#mse_ols = 
#results = [lambda_ridge, lambda_lasso, mse_ridge, mse_lasso, mse_ols]
#np.save(results, 'lab4_ex2_user.py')

#Comments:
#We did cross-validation using MSE as our performance metric. It's important 
#to note that we could easily do this for any other performance metric; the key 
#thing is that you should pick the performance metric based on the problem you want
#to solve. 
