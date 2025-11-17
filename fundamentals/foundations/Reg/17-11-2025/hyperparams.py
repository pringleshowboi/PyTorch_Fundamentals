import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Load the data set
cancer = load_breast_cancer()

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 19)

## Intiializaing model and dictionary of hyperparameters
lr = LogisticRegression(solver='liblinear', max_iter=1000)
parameters = {'penalty': ['l1', 'l2'], 'C': [1, 10, 100]}

##Setting up Grid Search
clf = GridSearchCV(lr, parameters)

## YOUR SOLUTION HERE ##
clf.fit(X_train, y_train)
best_model = clf.best_estimator_
print(best_model)
print(clf.best_params_)


## YOUR SOLUTION HERE ##
best_score = clf.best_score_
test_score = clf.score(X_test, y_test)
print(best_score)
print(test_score)

## YOUR SOLUTION HERE ##
hyperparameter_grid = pd.DataFrame(clf.cv_results_['params'])
grid_scores = pd.DataFrame(clf.cv_results_['mean_test_score'], columns = ['score'])

df = pd.concat([hyperparameter_grid, grid_scores], axis = 1)
print(df)