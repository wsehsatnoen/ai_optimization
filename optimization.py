import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV
from scipy.stats import loguniform
from sklearn.metrics import mean_squared_error, r2_score



# Optimization techniques

# Hyperparameter tuning using RandomizedSearchCV
def randomsearch(model, X_train, y_train, X_test, y_test):
    randomsearch = RandomizedSearchCV(model, param_distributions= {'tol': loguniform(1e-20, 1.0)}, cv=50)
    randomsearch.fit(X_train, y_train)

    y_pred = randomsearch.predict(X_test)

    print("Random Linear Regression Results")
    print("R2 Score: ", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].scatter(range(len(y_test)), y_test, color='blue', label='Actual Data')
    ax[0].plot(y_pred, color='orange', linewidth=3, label='Model Prediction')
    ax[0].set(xlabel='Data point', ylabel='Target', title='Actual data compared with model prediction')
    ax[0].legend()

    ax[1].scatter(y_pred, y_test, color='red')
    ax[1].set(xlabel='Predicted', ylabel='Actual', title='Predicted values vs Actual values')

    fig.show()

def halvesearch(model, X_train, y_train, X_test, y_test):
    halvesearch = HalvingRandomSearchCV(model, param_distributions= {'tol': loguniform(1e-20, 1.0)}, factor=3, resource='n_samples', max_resources='auto', random_state=42)
    halvesearch.fit(X_train, y_train)

    y_pred = halvesearch.predict(X_test)

    print("Halving Linear Regression Results")
    print("R2 Score: ", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].scatter(range(len(y_test)), y_test, color='blue', label='Actual Data')
    ax[0].plot(y_pred, color='orange', linewidth=3, label='Model Prediction')
    ax[0].set(xlabel='Data point', ylabel='Target', title='Actual data compared with model prediction')
    ax[0].legend()

    ax[1].scatter(y_pred, y_test, color='red')
    ax[1].set(xlabel='Predicted', ylabel='Actual', title='Predicted values vs Actual values')

    fig.show()