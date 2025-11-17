import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.io import arff

data = arff.loadarff('bone-marrow.arff')
df = pd.DataFrame(data[0]) # Access the data part of the tuple returned by loadarff

# *** FIX: Strip whitespace from column names to handle potential spacing issues in the ARFF file ***
df.columns = df.columns.str.strip()

# Check if 'Disease' column exists before dropping (optional debugging step)
if 'Disease' in df.columns:
    df.drop(columns=['Disease'], inplace=True)
else:
    print("Column 'Disease' not found, skipping drop operation.")


#Convert all columns to numeric, coerce errors to null values
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    
#Make sure binary columns are encoded as 0 and 1
for c in df.columns[df.nunique()==2]:
    df[c] = (df[c]==1)*1.0

# 1. Calculate the number of unique values for each column
print('Count of unique values in each column:')
print(df.nunique())
# 2. Set target, survival_status,as y; features (dropping survival status and time) as X
X = df.drop(columns=['survival_time','survival_status'])
y=df.survival_status
# 3. Define lists of numeric and categorical columns based on number of unique values
num_cols = X.columns[X.nunique()>7]
cat_cols = X.columns[X.nunique()<=7] # Defined cat_cols

# 4. Print columns with missing values
print("Columns with missing values:", X.columns[X.isnull().sum()>0].tolist())

# 5. Split data into train/test split
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=.2)

# 6. Create categorical preprocessing pipeline
# Using mode to fill in missing values and OHE
# *** FIX: Use 'sparse=False' for compatibility with older scikit-learn versions ***
cat_vals = Pipeline([("imputer",SimpleImputer(strategy='most_frequent')), ("ohe",OneHotEncoder(sparse=False, drop='first', handle_unknown = 'ignore'))])

# 7. Create numerical preprocessing pipeline
# Using mean to fill in missing values and standard scaling of features
num_vals = Pipeline([("imputer",SimpleImputer(strategy='mean')), ("scale",StandardScaler())])

# 8. Create column transformer that will preprocess the numerical and categorical features separately
preprocess = ColumnTransformer( transformers=[ ("cat_process", cat_vals, cat_cols), ("num_process", num_vals, num_cols) ] )
# 9. Create a pipeline with preprocess, PCA, and a logistic regresssion model
pipeline = Pipeline([("preprocess",preprocess), ("pca", PCA()), ("clf",LogisticRegression(max_iter=1000, solver='liblinear'))]) # Added max_iter and solver for convergence

# 10. Fit the pipeline on the training data (optional before grid search)
pipeline.fit(x_train, y_train)

# Predict the pipeline on the test data
initial_accuracy = pipeline.score(x_test,y_test)
print(f"\nInitial model accuracy: {initial_accuracy:.4f}\n")


# 11. Define search space of hyperparameters
# *** Update search_space as requested by original prompt ***
search_space = [
    {
        'pca__n_components': np.linspace(5, 35, 7).astype(int), # 5, 10, 15, 20, 25, 30, 35
        'clf': [LogisticRegression(max_iter=1000, solver='liblinear')],
        'clf__C': np.logspace(-4, 4, 4),
    },
    {
        'pca__n_components': np.linspace(5, 35, 7).astype(int),
        'clf': [RandomForestClassifier(random_state=1)],
        'clf__n_estimators': [50, 100, 200], # Added values for the parameter
    }
]

# 12. Search over the hyperparameters in search_space for the pipeline using GridSearchCV. Fit on the training set.
grid_search = GridSearchCV(pipeline, search_space, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
grid_search.fit(x_train, y_train)

# 13. Save the best estimator from the gridsearch and print attributes and final accuracy on test set
best_model = grid_search.best_estimator_

# 14. Print attributes of best_model
print("\n--- Best Model Attributes ---")
print(f"Classifier Type: {type(best_model.named_steps['clf']).__name__}")
print(f"PCA Components Selected: {best_model.named_steps['pca'].n_components_}")
print(f"Classifier Hyperparameters: {best_model.named_steps['clf'].get_params()}")
print(f"Best Score (CV Accuracy): {grid_search.best_score_:.4f}")

# 15. Print final accuracy score
final_accuracy = best_model.score(x_test, y_test)
print("\n--- Model Evaluation ---")
print(f"Initial model accuracy: {initial_accuracy:.4f}")
print(f"Best model accuracy on test set: {final_accuracy:.4f}")
print(f"Accuracy improvement: {(final_accuracy - initial_accuracy):.4f}")