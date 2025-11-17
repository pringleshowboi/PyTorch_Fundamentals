import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns
#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

#1. Create a pipeline `num_vals` to process numerical data

num_vals = Pipeline([("imputer",SimpleImputer()), ("scale",StandardScaler())])

#2. Create a pipeline `cat_vals` to process categorical data
cat_vals = Pipeline([("imputer",SimpleImputer(strategy = 'most_frequent')), ("ohe",OneHotEncoder(drop = 'first', sparse = False))])


#3. Create a column transformer, `preprocess` with the numerical and categorical pipelines
preprocess = ColumnTransformer(
    transformers=[
        ("num_preprocess", num_vals, num_cols),
        ("cat_preprocess", cat_vals, cat_cols)      
    ]
)


#4. Fit the preprocess transformer to training data
preprocess.fit(x_train)
#Transform the test data
x_transform = preprocess.transform(x_test)




