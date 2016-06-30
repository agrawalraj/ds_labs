

"""
Lab : Solutions 
"""

import numpy as np 

from sklearn import decisionTrees
from sklearn import datasets
from sklearn.utils import shuffle

class additiveBinaryFunctions:
    """
    function: lambda function that takes two inputs 
    ex. 
    >> squared_loss = lambda x, y: (x - y) ** 2
    >> sum_func = lambda x, y: x + y
    >> iterative_function = additiveBinaryFunctions(squared_loss)
    >> iterative_function(3, 4) #1
    >> iterative_function + sum_func
    >> iterative_function(3, 4) #8 = (3 - 4)^2 + (3 + 4)
    """

    def __init__(self, function):
        self.function_list = []
        self.function_list.append(function)

    def __add__(self, function):
        self.function_list.append(function)

    def __call__(self, x, y): 
        total = sum([f(x, y) for f in self.function_list])
        return total 
     

def KLDerivative():


def squareLossDerivative(X, Y):
    return Y - X

def functionGradient():

def pseudoResiduals():
    functionGradient(Y, X, loss)

def initModel():


def solveMultiplier():



def gradientBoosting(Y, X, num_iter):
    curr_model = initModel()
    while i < num_iter:
        i += 1 
        pseudo_residuals = -1 * functionGradient(Y, X, curr_model)

        #Fit base learner to pseudo residuals 
        weak_learner = model()
        weak_learner.fit(X, pseudo_residuals)

        #Find out how much to weight this function 
        weak_learner_multiplier = solveMultiplier()

        #Update model  
        curr_model = curr_model + weak_learner_multiplier * weak_learner 


#Gradient boosting for regression 

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=42)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

#Gradient boosting for classification 

squared_loss = lambda x, y: .5 * (x - y) ** 2
KL_loss = lambda x, y: .5 * (x - y) ** 2




