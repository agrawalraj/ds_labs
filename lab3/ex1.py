
# Author: Raj Agrawal 
# Material: Adapted fom Mike Bernico's CS 570 Class 

"""
Lab 3: Converting Unstructured Data to Features: LSA

In this lab we will do LSA on a group of newsgroup posts from the newsgroup
'rec.sport.baseball'

We started you off with the libraries we used to do this lab. You are welcome to 
use other libraries but make sure the output data types are correct (we will
specify what data type is needed later). If you get stuck take a look at the link 
in 'References'.

When you're done run your script $python ex1.py on your terminal and then
to receive feedback run $python lab3_ex1_grader.py. Make 
sure you are in the directory where this script is located.  

References: https://github.com/mbernico/CS570/blob/master/LSA%20Text.ipynb

"""

# Import Libraries 
import random
import nltk
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# nltk.download('stopwords') - Uncomment if you have not run this before 

# Set seed for consistent results 
SEED_INT = 42
random.seed(SEED_INT)

# First get the newsgroup Data
categories = ['rec.sport.baseball']
dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=SEED_INT, 
	                         categories=categories)

corpus = dataset.data #List of 994 docs 

#Stop words - Later you add more to this set
stopset = set(stopwords.words('english')) 

################ Fit LDA Model ##########################

# Most of this is just to check that you can fit an LSA Model - The input
# paramters are not really good for this dataset (need consistency to run 
# autograder). After you finish this you should play add more words to the 
# stopset and play around with some of the hyperparameters to get a better 
# fit 

# YOUR CODE HERE - Uncomment and fill in variables below 

# corpus = Lowercase all letters in the corpus above 

# vectorizer = Set up object to convert corpus to a matrix of TF-IDF features
#              use inverse-document-frequency reweighting
#              set the ngram range from 1 to 3 inclusive 

# X = Convert corpus to a matrix of TF-IDF features using the vectorizer 

# lsa =  Set up using TruncatedSVD w/ n_components = 1 and 100 iterations 
#        set random_state = SEED_INT 

# Fit lsa object on X

# top10 = Put the top 10 terms of concept 0 into a list along with term frequency 
#         sorted high to low e.g. [('cat', .218), ('dog', .195), ...]

# Saves top10 list for autograder 
# np.save('../temp_saved_work/lab3_ex1_user', top10)

########### Ungraded Exploration ######################

# Not graded but try changing the stopset by adding more terms specific to 
# the data set and altering some of the hyperparamters of the LSA model 
