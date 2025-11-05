import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())
prod_per_year = df.groupby('year', as_index=False)['totalprod'].mean()
X = prod_per_year['year']
y = prod_per_year['totalprod']
X = X.values.reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)
y_predict = regr.predict(X)

plt.scatter(y, X)
plt.plot(y_predict, X)
plt.show()

nums = np.array(range(1, 11))
X_future = np.array(range(2018, 2028)).reshape(-1, 1)
future_predict = regr.predict(X_future)
plt.plot(future_predict, X_future)
