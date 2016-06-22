
#Author: Raj Agrawal 

"""
Lab 4: Solutions 
"""

import numpy as np 

def ridgeGradient(Y, X, beta, lam):
    """
	References:
	"""
	#Residuals 
    R = Y - X.dot(beta)
    R = R.T

    #First l2 derivative part
    n = len(Y) 
    l2_deriv_first = (-2 / n) * X.T.dot(R)

    #Second l2 derivative part 
    l2_deriv_second = 2 * lam * beta 
    return l2_deriv_first + l2_deriv_second

def gradientRidgeDecent(Y, X, lam, gamma=.01, n_iter=30, beta_init=0):
	"""
	
	"""
	if n_iter == 0:
    	return beta_init
    else:
    	beta_update = beta - gamma * ridgeGradient(Y, X, beta)
    	return gradientLassoDecent(Y, X, lam, gamma, n_iter - 1, beta_update)

def ridgeRegression(Y, X, lam):
	return gradientRidgeDecent(Y, X, lam)

X = np.array([1, 2, 3, 1.1, 2.2, 3.3, .9, .9, 2.1, 2.7])
X.shape = (3, 3)
theta_true = np.array([4, 5, 6])
noise = np.array([.1, .8, .6])
Y = X.dot(theta_true) + noise
lambda_ridge = 3

beta_ridge = ridgeRegression(Y, X, lam)
np.save(beta_ridge, 'lab4_ex1_sol.py')
