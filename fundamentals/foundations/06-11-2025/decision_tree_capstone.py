import codecademylib3
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

# variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

# Print number of countries by landmass, or continent
print(df['landmass'].value_counts().astype(int))
# Create a new dataframe with only flags from Europe and Oceania
df_36 = df[(df['landmass'] == 3) | (df['landmass'] == 6)]

# Print the average vales of the predictors for Europe and Oceania
print(df_36[var].groupby(df_36['landmass']).mean())

# Create labels for only Europe and Oceania
labels = df_36['landmass']

# Print the variable types for the predictors
print(df_36[var].dtypes)
labels = (df["landmass"].isin([3,6]))*1

# Create dummy variables for categorical predictors
data = pd.get_dummies(df[var].astype(bool))

# Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=1, test_size=0.4)

# Fit a decision tree for max_depth values 1-20
acc_depth = []
for depth in range(1,21):
  dtree = DecisionTreeClassifier(max_depth=depth)
  dtree.fit(x_train, y_train)
  acc_depth.append(dtree.score(x_test,y_test))

depths = list(range(1, 21))
plt.figure()
plt.plot(depths, acc_depth)
plt.title("Accuracy vs. Max Depth")
plt.show()

max_accuracy = np.max(acc_depth)
print(f"Maximum accuracy (Max Depth): {max_accuracy}")

# Find the depth that gave the maximum accuracy (there might be more than one, pick the first one)
optimal_depth = depths[np.argmax(acc_depth)]

# Re-fit a decision tree with the optimal depth for plotting (optional, the code below focuses on ccp)
# dtree_optimal_depth = DecisionTreeClassifier(max_depth=optimal_depth)
# dtree_optimal_depth.fit(x_train, y_train)


# --- Cost-Complexity Pruning Implementation ---

# First, train a full, unpruned tree to get the effective alphas
dtree_full = DecisionTreeClassifier(random_state=0)
path = dtree_full.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Remove the largest alpha that corresponds to the trivial tree (single node)
ccp_alphas = ccp_alphas[:-1] 

# Train a tree for each alpha value
acc_pruned = []
for ccp_alpha in ccp_alphas:
    dtree = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=0)
    dtree.fit(x_train, y_train)
    acc_pruned.append(dtree.score(x_test, y_test))

# Plot accuracy vs ccp_alpha
plt.figure()
plt.plot(ccp_alphas, acc_pruned, marker='o', drawstyle="steps-post")
plt.xlabel("alpha")
plt.ylabel("test set accuracy")
plt.title("Accuracy vs. alpha for training and test sets")
plt.show()

max_ccp_accuracy = np.max(acc_pruned)
print(f"Maximum accuracy (CCP Alpha): {max_ccp_accuracy}")

# Find the optimal alpha
optimal_ccp_alpha = ccp_alphas[np.argmax(acc_pruned)]

# Re-fit a decision tree with the optimal ccp_alpha
dtree_optimal_ccp = DecisionTreeClassifier(ccp_alpha=optimal_ccp_alpha, random_state=0)
dtree_optimal_ccp.fit(x_train, y_train)

# Plot the optimally pruned decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(dtree_optimal_ccp, 
               feature_names=data.columns.tolist(),
               class_names=['Other', 'Europe/Oceania'],
               filled=True,
               rounded=True,
               precision=2,
               fontsize=8)
plt.title(f"Optimally Pruned Decision Tree (Alpha={optimal_ccp_alpha:.4f})")
plt.show()