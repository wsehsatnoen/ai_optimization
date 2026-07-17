import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import loguniform

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
# and one to show a direct comparison of the predicted values against the actual values.
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].scatter(range(len(y_test)), y_test, color='blue', label='Actual Data')
ax[0].plot(y_pred, color='orange', linewidth=3, label='Model Prediction')
ax[0].set(xlabel='Data point', ylabel='Target', title='Actual data compared with model prediction')
ax[0].legend()

ax[1].scatter(y_pred, y_test, color='red')
ax[1].set(xlabel='Predicted', ylabel='Actual', title='Predicted values vs Actual values')

fig.show()

# Optimization techniques

# Hyperparameter tuning using RandomizedSearchCV
randomsearch = RandomizedSearchCV(linreg, param_distributions={'fit_intercept': [True, False], 'tol': loguniform(1e-20, 1.0)}, cv=50)
randomsearch.fit(X_train, y_train)

y_pred = randomsearch.predict(X_test)

print("Grid Linear Regression Results")
print("R2 Score: ", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 5))

ax2[0].scatter(range(len(y_test)), y_test, color='blue', label='Actual Data')
ax2[0].plot(y_pred, color='orange', linewidth=3, label='Model Prediction')
ax2[0].set(xlabel='Data point', ylabel='Target', title='Actual data compared with model prediction')
ax2[0].legend()

ax2[1].scatter(y_pred, y_test, color='red')
ax2[1].set(xlabel='Predicted', ylabel='Actual', title='Predicted values vs Actual values')

fig2.show()

# Halving search using HalvingRandomSearchCV
halvedge = HalvingRandomSearchCV(linreg, param_distributions={'fit_intercept': [True, False], 'tol': loguniform(1e-20, 1.0)}, cv=50)
halvedge.fit(X_train, y_train)

print("Halving Linear Regression Results")
print("R2 Score: ", r2_score(y_test, halvedge.predict(X_test)))
print("Mean Squared Error:", mean_squared_error(y_test, halvedge.predict(X_test)))

fig3, ax3 = plt.subplots(ncols=2, figsize=(10, 5))

ax3[0].scatter(range(len(y_test)), y_test, color='blue', label='Actual Data')
ax3[0].plot(halvedge.predict(X_test), color='orange', linewidth=3, label='Model Prediction')
ax3[0].set(xlabel='Data point', ylabel='Target', title='Actual data compared with model prediction')
ax3[0].legend()

ax3[1].scatter(halvedge.predict(X_test), y_test, color='red')
ax3[1].set(xlabel='Predicted', ylabel='Actual', title='Predicted values vs Actual values')

fig3.show()






