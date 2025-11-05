import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis = pd.read_csv('tennis_stats.csv')
print(tennis.head())
features = tennis[['FirstServeReturnPointsWon']]
outcome = tennis[['Winnings']]

# perform exploratory analysis here:
plt.scatter(tennis[['BreakPointsOpportunities']], tennis[['Winnings']])
x_train, x_test, y_train, y_test = train_test_split(tennis[['BreakPointsOpportunities']], tennis[['Winnings']], train_size=0.8, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)
prediction = model.predict(y_test)
plt.scatter(y_test, prediction, alpha=0.4)
plt.show()
