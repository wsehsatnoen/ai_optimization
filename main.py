from dataprep import X_train, y_train, X_test, y_test

from linreg import linreg
from optimization import randomsearch, halvesearch
from regularization import lasso, ridge
from ensemble import bagger, ada

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

# Original Model
linreg(linear_model, X_train, y_train, X_test, y_test)

# Optimized Models
# randomsearch(linear_model, X_train, y_train, X_test, y_test)
# halvesearch(linear_model, X_train, y_train, X_test, y_test)

# Regularized Models
# lasso(X_train, y_train, X_test, y_test)
# ridge(X_train, y_train, X_test, y_test)

# Ensemble Learning Techniques
bagger(linear_model, X_train, y_train, X_test, y_test)
ada(linear_model, X_train, y_train, X_test, y_test)










