
Author: Raj Agrawal 

"""
Lab 4: Ex2 Solutions
"""

import numpy as np

from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression 
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes['data']
Y = diabetes['target']

def kFoldCV(k, lam, model):


def MSE(prediction, actual):
	return sum((prediction - actual) ** 2)