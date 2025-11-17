import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
df = pd.read_csv('adult.data', header=None, names = col_names)

#Distribution of income


#Clean columns by stripping extra whitespace for columns of type "object"

#Create feature dataframe X with feature columns and dummy variables for categorical features
X = None
#Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greather than 50k
y = None

#Split data into a train and test set


#Instantiate random forest classifier, fit and score with default parameters


#Tune the hyperparameter max_depth over a range from 1-25, save scores for test and train set
np.random.seed(0)
accuracy_train=[]
accuracy_test = []


    
#Find the best accuracy and at what depth that occurs



#Plot the accuracy scores for the test and train set over the range of depth values  


#Save the best random forest model and save the feature importances in a dataframe
best_rf = RandomForestClassifier(max_depth=None)



#Create two new features, based on education and native country
# df['education_bin'] = None

# feature_cols = ['age',
#        'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race','education_bin']
#Use these two new additional features and recreate X and test/train split


#Find the best max depth now with the additional two features





#Save the best model and print the two features with the new feature set


