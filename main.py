import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Importing the dataset using Pandas
data = pd.read_csv('Dataset.csv')

# Here is where we are splitting up the dataset into features and a target.
features = data.iloc[0:1000, 0:-1]
target = data.iloc[0:1000, -1]

# Here we are taking the split dataset and creating our training and testing set.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Linear Regression Model implementation
linreg = LinearRegression()

# Training the model
linreg.fit(X_train, y_train)

# Testing the model
y_pred = linreg.predict(X_test)

# Displaying the results via command line
print("Linear Regression Results")
print("R2 Score: ", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Graphing the results on two plots, one to show model's prediction against the actual target,
# and one to show a direct comparision of the predicted values against the actual values.
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].scatter(range(len(y_test)), y_test, color='blue')
ax[0].plot(y_pred, color='orange', linewidth=3)
ax[0].set(xlabel='Data point', ylabel='Target', title='Actual vs Predicted')

ax[1].scatter(y_pred, y_test, color='red')
ax[1].set(xlabel='Predicted', ylabel='Actual', title='Predicted vs Actual')

fig.show()






