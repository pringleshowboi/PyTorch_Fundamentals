import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Load the data set
cancer = load_breast_cancer()

# Split the data into training and testing sets
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

## YOUR SOLUTION HERE ##

distributions = {'penalty': ['l1', 'l2'], 'C': uniform(loc=0, scale=100)}

## YOUR SOLUTION HERE ##
first_draw = distributions['C'].rvs(10)

second_draw = distributions['C'].rvs(10)
print(first_draw)
print(second_draw)


## YOUR SOLUTION HERE ##

# The logistic regression model
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000)

# Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr, distributions, n_iter=8)