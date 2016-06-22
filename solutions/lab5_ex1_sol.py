
#Author: Raj Agrawal 

"""
Lab 5: Ex1 Possible Solutions 

"""
import numpy as np 
import pandas as pd 
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.linear_model import LogisticRegression 

data = pd.read_csv('../data/titantic.csv')

#Seems like age, sex, Parch, SibSp could be predictive 
#Not clear at this point how to choose the variables 
#Pclass, Fare, and Embarked - these seem correlated 

############################## EDA ##############################

#Also see: https://github.com/mbernico/CS570/blob/master/titanic%20EDA.ipynb 

#Check for missing values 
sum(pd.isnull(data['Survived'])) #0 missing values
sum(pd.isnull(data['Age'])) #177 missing ages 
sum(pd.isnull(data['Sex'])) #0 missing values 
sum(pd.isnull(data['Pclass'])) #0 missing values
sum(pd.isnull(data['Parch'])) #0 missing values
sum(pd.isnull(data['SibSp'])) #0 missing values
sum(pd.isnull(data['Fare'])) #0 missing values 
sum(pd.isnull(data['Embarked'])) #2 missing values
sum(pd.isnull(data['Cabin'])) #687 - nearly whole dataset 

#Check associations between Pclass, Fare, and Embarked 
data.boxplot(column='Fare', by='Pclass') 
#Lot more variability in Fare price for class 1. Class 2 
#and 3 seem roughly the same. Class 1 (as expected) has its 
#25th% fare price near the 75%-tile for class 2 and 3 

data.boxplot(column='Fare', by='Embarked') 

pd.crosstab(index=data['Pclass'], columns=data['Embarked'], normalize='index')
#90% of class 2 passangers embarked from south hampton 

pd.crosstab(index=data['Survived'], columns=data['Embarked'], normalize='columns')
#This says 55% of those embarking from Chesiwick survived, etc. 

data.boxplot(column='Age', by='Survived') 

# One-Hot Encoding 
data['Sex'] = pd.get_dummies(data['Sex'])['male']
encoded = pd.get_dummies(data['Embarked'])
data['C'] = encoded['C']
data['S'] = encoded['S']
data.drop('Embarked', axis=1, inplace=True) #Redundant information 
data.drop('PassengerId', axis=1, inplace=True)

########################### Missing Age ###########################

#One possible way to fill in the missing 177 age values is predicting 
#age by the other variables

unknown_ages_bool = pd.isnull(data['Age'])
missing_ages_data = data[unknown_ages_bool]

have_ages_bool = [not logical for logical in unknown_ages_bool]
have_ages_data = data[have_ages_bool]

#Here just act like our dataset is 'have_ages_data' and our goal is to create
#a model to predict age. Once we have this model, we can just feed in the 
#missing_ages_data to get predicted ages 

#Since age is continuous this is a Regression problem 

#We chose linear regression as our model. Other machine learning techniques 
#might work better but this is what we've covered so far. 

#Shuffle data
have_ages_data = have_ages_data.reindex(np.random.permutation(have_ages_data.index))

Y_age = have_ages_data['Age']
X_age = have_ages_data.drop('Age', axis=1, inplace=False)

#Remove out unwanted columns from design matrix X_age
X_age.drop('Name', axis=1, inplace=True) #potentially useful in predicting ethnicity
                                         #but a lot more work 

X_age.drop('Cabin', axis=1, inplace=True) #Too many missing values 
X_age.drop('Ticket', axis=1, inplace=True) #714 rows 


X_age_train = X_age[0:500]
X_age_test = X_age[500:714]

Y_age_train = Y_age[0:500]
Y_age_test = Y_age[500:714]

#Fit different esimators and pick the one that performs best 
#Here really does not make sense to use Ridge Regression or Lasso  
#We fit all three anyways because can apply same idea later when 
#it is relevant 

ESTIMATORS = {
	"Linear regression": LinearRegression(), 
	"Ridge": RidgeCV(), 
	"Lasso": LassoCV() 
}

def rmse(predictions, actual):
	return (sum((predictions - actual) ** 2) / len(predictions)) ** .5

y_test_predict = dict()
for name, esimator in ESTIMATORS.items():
	esimator.fit(X_age_train, Y_age_train)
	y_test_predict[name] = rmse(Y_age_test, esimator.predict(X_age_test))

#12 years RMSE is the smallest, corresponding to linear regression  
final_age_model = LinearRegression()
final_age_model.fit(X_age, Y_age)

#Get rid of non-numeric data since o/w errors for predict function 
subset_missing_data = missing_ages_data.drop('Age', axis=1, inplace=False)
subset_missing_data.drop('Cabin', axis=1, inplace=True)
subset_missing_data.drop('Ticket', axis=1, inplace=True)
subset_missing_data.drop('Name', axis=1, inplace=True)

missing_age_predictions = final_age_model.predict(subset_missing_data)

#Update Age column 
missing_ages_data['Age'] = missing_age_predictions

########################### Fit Logistic Model ###########################

final_data = have_ages_data.append(missing_ages_data, ignore_index=True)
Y = final_data['Survived']

final_data.drop('Cabin', axis=1, inplace=True)
final_data.drop('Ticket', axis=1, inplace=True)
final_data.drop('Name', axis=1, inplace=True)
final_data.drop('Survived', axis=1, inplace=True)

#TODO: Pclass is categorical 

logit_model = LogisticRegression(penalty='l2', C=1)
logit_model.fit(final_data, Y)
