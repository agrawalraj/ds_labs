
#Author: Raj Agrawal 

"""
Lab 4: Lasso and Ridge Regression

In this lab you will implement Ridge Regression without using sklearn. 
You can also try implementing Lasso at the endbut this will not be graded. 

References:
See notes for all references.
"""

import numpy as np 
 
def ridgeGradient(Y, X, beta, lam):
    """
    This calculates the gradient of the ridge cost function at beta. 
    
    returns:
        gradient: a p x 1 numpy array 
    
    Note:
        If you have trouble calculating gradient look at solutions (easy to 
        to forget some of calculus). 
    
	"""
	return None 

def gradientRidgeDecent(Y, X, lam, gamma=.01, n_iter=30, beta_init=0):
	"""
    This does gradient decent in solve the ridge regression optimization 
    problem 
	"""
	return None 

def ridgeRegression(Y, X, lam):
	"""
    This solves the optimization problem: 
    beta_lasso = arg min 1/n * (||(Y - X*beta)||_2)^2 + lambda * ||beta||_2, where
    the min is taken over beta in R^p and || ||_2 denotes the L2 norm using 
    gradient decent.  
	
	input:
	    Y: Numpy n x 1 array consisting of n response values
	    X: Numpy n x p array. Rows = observations. Columns = Measurements 
	    lambda: Tuning parameter used in the L1 regularization
	
	returns:
	    beta: Returns the lasso fitted beta coefficient  

    Note: 
        There is a closed-form linear algebra solution for ridge regression but 
        using gradient decent is usually faster for larger problems. This is 
        because matrix inversion can be expensive if data dimensionality p is large 
        (which is required in closed-form solution).

        Here don't worry about intercept term: We assume that this has already 
        been taken care for us in the design matrix X (e.g. by adding columns of 
        just 1's or recentering). In other words, just solve the optimzation 
        problem above. 

	"""
	#YOUR CODE HERE 
	return None 

#Try testing a few toy examples out on your functions 
X = np.array([1, 2, 3, 1.1, 2.2, 3.3, .9, .9, 2.1, 2.7])
X.shape = (3, 3)
theta_true = np.array([4, 5, 6])
noise = np.array([.1, .8, .6])
Y = X.dot(theta_true) + noise
lambda_ridge = 3

beta_ridge = ridgeRegression(Y, X, lam)
np.save(beta_ridge, 'lab4_ex1_user')

########################## Optional Ex. ##########################

def lassoRegression(Y, X, lambda):
	
	"""
    This solves the optimization problem 
    beta_lasso = arg min 1/n * (||(Y - X*beta)||_2)^2 + lambda * ||beta||_1, where
    the min is taken over beta in R^p and || ||_2 denotes L2 norm and || ||_1
    the L1 norm   
	
	input:
	    Y: Numpy n x 1 array consisting of n response values
	    X: Numpy n x p array. Rows = observations. Columns = Measurements 
	    lambda: Tuning parameter used in the L1 regularization
	
	returns:
	    beta: Returns the lasso fitted beta coefficient  

    Note: 
        Here don't worry about intercept term: We assume that this has already 
        been taken care of in the design matrix X (e.g. by adding columns of 
        just 1's or by recentering). In other words, just solve the optimzation 
        problem above. 

        Trickier here b/c L1 penalty is not differentiable so can't use gradient
	    decent automatically. There is sub-gradient decent but not that fast. You
	    can try searching Google for ways to solve this. 

	"""
	#YOUR CODE HERE
