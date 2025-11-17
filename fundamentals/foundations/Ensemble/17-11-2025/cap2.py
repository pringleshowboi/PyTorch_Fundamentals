import pandas as pd
import numpy as np
import codecademylib3

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

path_to_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

col_names = [
    'age', 'workclass', 'fnlwgt','education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain','capital-loss',
    'hours-per-week','native-country', 'income'
]

df = pd.read_csv(path_to_data, header=None, names = col_names)
print(df.head())

#Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()

target_column = "income"
raw_feature_cols = [
    'age',
    'education-num',
    'workclass',
    'hours-per-week',
    'sex',
    'race'
]

##1. Percentage of samples with income < and > 50k

##2. Data types of features

##3. Preparing the features

##4. Convert target variable to binary

##5a. Create train-est split

##5b. Create base estimator and store it as decision_stump


##6. Create AdaBoost Classifier

##7. Create GradientBoost Classifier


##8a.Fit models and get predictions

##8b. Print accuracy and F1



##9. Hyperparameter Tuning
n_estimators_list = [10, 30, 50, 70, 90]
from sklearn.model_selection import GridSearchCV


##10. Plot mean test scores
#ada_scores_list 