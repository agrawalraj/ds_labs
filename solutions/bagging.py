
"""
Lab : Ex 2 Solutions

"""

import bootstrap 
from sklearn import decisionTrees
from sklearn import tree 
from collections import Counter

def mostCommon(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def treeBagging(Y, X, num_resamples=1000):
    n = len(Y)
    bootstrap_sample_indcs = bootstrap.bootstrapSample(n, num_resamples)
    models = []
    for i in range(n):
        rows = bootstrap_sample_indcs[i]
        X_sample = X[rows]
        Y_sample = Y[rows]
        clf = tree.DecisionTreeClassifier() 
        clf = clf.fit(X_sample, Y_sample)
        models.append(clf)
    return models 
    
#Nearly identical to treeBagging function 
def randomForest(Y, X, prop, num_resamples=1000):
    n = len(Y)
    bootstrap_sample_indcs = bootstrap.bootstrapSample(n, num_resamples)
    models = []
    indcs = np.arange(n)
    num_features_to_keep = round(n * prop)
    for i in range(n):
        rows = bootstrap_sample_indcs[i]
        X_sample = X[rows]
        np.random.shuffle(indcs)
        X_sample_subset = X_sample[indcs[1:num_features_to_keep]]
        Y_sample = Y[rows]
        clf = tree.DecisionTreeClassifier() 
        clf = clf.fit(X_sample_subset, Y_sample)
        models.append(clf)
     return models 

def treePredict(model_lst, new_data_point):
    predictions = []
    for model in model_lst:
        predictions.append(model.predict(new_data_point))

    #Majority vote wins 
    return most_common(predictions)


#Do stuff with sample dataset 

