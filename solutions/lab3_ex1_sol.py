
# Author: Raj Agrawal 
# Material: Adapted fom Mike Bernico's CS 570 Class 

"""
Lab 3: Converting Unstructured Data to Features: LSA

Solutions 

"""

# Import Libraries 
import random
import nltk

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

#Stop words - Later you will add more to this set
stopset = set(stopwords.words('english')) 

corpus = [doc.lower() for doc in corpus]
vectorizer = TfidfVectorizer(stop_words=stopset, use_idf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(corpus)

lsa = TruncatedSVD(n_components=1, n_iter=100, random_state=42)
lsa.fit(X)

terms = vectorizer.get_feature_names()
comp0 = lsa.components_[0]
termsInComp = zip(terms,comp)
top10 = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]

np.save('../temp_saved_work/lab3_ex1_sol', top10)
