
"""
Lab : Ex 1 Solutions  

References: 
See http://www.sagepub.com/sites/default/files/upm-binaries/21122_Chapter_21.pdf
on pg. 597 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 

def bootstrapSample(n, num_resamples=1000):
    """
    input:
        n - number of data points   
        num_resamples - number of bootstrapped samples to use. Defaults to 1000. 

    returns:
        bootstrap_sample_indcs - num_resamples x n numpy array. Each row corresponds to 
                           the n row indcs to randomly choose in the data matrix 
    """
    bootstrap_sample_indcs = []
    indcs = np.arange(n)

    for i in range(num_resamples):
        sample_indcs = np.random.randint(0, n, n)
        bootstrap_sample_indcs.append(sample_indcs)
    
    return np.array(bootstrap_sample_indcs)

def bootstrapRegression(Y, X, num_resamples=1000):
    n = len(Y)
    bootstrapped_indcs = bootstrapSample(n, num_resamples)

    #First get regression coefficients for orginal dataset 
    lmodel = LinearRegression()
    lmodel.fit(Y, X)
    true_insamp_coef = lmodel.coef_
    Y_hat = lmodel.fitted
    true_insamp_residuals = Y - Y_hat 
    bootstrapped_coef = []
    for i in range(n):
        row = bootstrapped_indcs[i, ]
        sampled_residuals = true_insamp_residuals[row]
        sampled_Y = Y_hat + sampled_residuals
        lmodel_samp = LinearRegression()
        lmodel_samp.fit(sampled_Y, X)
        bootstrapped_coef.append(lmodel_samp.coef_)
    return bootstrapped_coef

def bootstrappedMeanEstimator(X, num_resamples=1000):
    n = len(X)
    indcs = np.arange(n)
    bootstrap_sample_indcs = bootstrapSample(n, num_resamples)
    bootstrapped_means = []
    for row in bootstrap_sample_indcs:
        samp_mean = np.mean(X[row])
        bootstrapped_means.append(samp_mean)
    return bootstrapped_means

#Boostrap mean

#Suppose the true mean is u = 10 and our n samples come iid from a 
#N(u, 1). Then we know the sampling distribution of the mean should
#be a N(u, 1/n) since 1/n * sum x_i = 1/n * N(n*u, n) equals in 
#distribution to a N(u, 1/n). Hopefully, we will roughly see a normal 
#w/ var = 1/n in the histogram of the bootstrap samples 

u = 10
n = 10000
X = np.random.normal(loc=u, scale=1, size=n)

plt.hist(X) #This looks really close to a N(u, 1). This is key because in the 
            #the bootrap we are essentially "plugging in" X as our resulting 
            #distribution. See notes for more details. 

boot_means = bootstrappedMeanEstimator(X)
plt.hist(boot_means) #Looks really close to N(0, 1/n)
boot_mean = np.mean(boot_means)
boot_var = np.var(boot_means) #This is really close to 1/n 

#Bootstrap Confidence Intervals for "unknown" u 

boot_mean_ci = (boot_mean - boot_var, boot_mean + boot_var)

#np.save(boot_mean_ci)

#Boostrap linear regression toy example 

X = np.arange(1000)
Y = 2 * X + np.random.normal(loc=0, scale = 3, size=1000)

# lmodel = LinearRegression()
# lmodel.fit(X, Y)
# true_insamp_coef = lmodel.coef_











   

