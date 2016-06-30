
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
from scipy.fftpack import dct, idct 
from scipy.stats import threshold
from numpy.random import randn 
from sklearn.linear_model import LassoCV, Lasso
from cvxopt import matrix, mul, div, cos, sin, exp, sqrt
from cvxopt import blas, lapack, solvers


def compress(fraction):

def decompress():


#Load in image 
tiger_image = mpimg.imread('tiger.jpg')
tiger_image = tiger_image[:,:,0] #Take one color channel 

#See how it looks 
tiger_image.shape #1024 x 1024 or 1,048,576 pixels 
imgplot = plt.imshow(tiger_image[:,:,0])
#plt.show()

#Let's see how we can compress the image and then recover it 
#this requires taking wavelet transform to get sparse representation 
#of image 

cosine_transform = dct(tiger_image)
plt.imshow(cosine_transform)
#plt.show()

#This transfor is 1 - 1 that is we can get back the original image 
#by using inverse cosine transform 
original = idct(cosine_transform)
#plt.imshow(original) #Eactly the same 

#Take only the top 100k values (out of ~ 1 million pixels) in the 
#transformed matrix 
vectorized_dct_image = dct(tiger_image)
vectorized_dct_image.shape = (1024 * 1024, ) 
sorted(abs(vectorized_dct_image))[900000] #pick pixels greater than 1247.4 in abs value 
max(abs(vectorized_dct_image)) #max value 292024

#Get sampling matrix 
top100k = threshold(abs(vectorized_dct_image), 1247.4, 292024, 0) #set 0's 
top100k = threshold(vectorized_dct_image, 0, 0, 1) #set everything else to 1 

sub_sampled_signal = top100k.dot(vectorized_dct_image)


#Random array 
old_man = mpimg.imread('old_man.jpg')
old_man = old_man[:, :, 0]
transform_old_man = dct(old_man)
transform_old_man.shape = (225 ** 2, )
vectorized_image = transform_old_man

A = randn(15000, 225 * 225)

sub_sampled_image = A.dot(vectorized_image)


LassoCV = LassoCV()

recovered_dct_image = LassoCV.fit(A, sub_sampled_image)



old_man = mpimg.imread('dog.jpg')
old_man = old_man[:, :, 0]
old_man.shape = (128 ** 2, )
plt.plot(old_man) #No sparsity 

old_man.shape = (128, 128)
plt.imshow(old_man)
transform_old_man = dct(old_man)
transform_old_man.shape = (128 ** 2, )
vectorized_image = transform_old_man
plt.plot(vectorized_image) #Sparsity -> ready for Lasso 

A = randn(5000, 128 * 128)

sub_sampled_image = A.dot(vectorized_image)


LassoCV = LassoCV()

recovered_dct_image = LassoCV.fit(A, sub_sampled_image)

recovered_dft_values = recovered_dct_image.coef_

recovered_dft_values.shape = (128, 128)

recovered_image = idct(recovered_dft_values)
plt.imshow(recovered_image)

las = Lasso(alpha=1174)
recovered_dct_image2 = las.fit(A, sub_sampled_image)

recovered_dft_values2 = recovered_dct_image2.coef_
recovered_dft_values2.shape = (128, 128)
recovered_image2 = idct(recovered_dft_values2)
plt.imshow(recovered_image2)


#Put crappy matrix e.g. doesnt satisfy RIP and check if no recovery 




