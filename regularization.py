from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def lasso(X_train, y_train, X_test, y_test):
    lasso = Lasso()
    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test)

    print("Lasso Results")
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

def ridge(X_train, y_train, X_test, y_test):
    ridge = Ridge()
    ridge.fit(X_train, y_train)

    y_pred = ridge.predict(X_test)

    print("Ridge Results")
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

