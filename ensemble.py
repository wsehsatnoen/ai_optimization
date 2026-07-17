from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def bagger(model, X_train, y_train, X_test, y_test):
    bgr = BaggingRegressor(model, n_estimators=100, max_samples=0.5, random_state=0)
    bgr.fit(X_train, y_train)

    y_pred = bgr.predict(X_test)

    print("Bagging Ensemble Results")
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

def ada(model, X_train, y_train, X_test, y_test):
    ada = AdaBoostRegressor(model, n_estimators=100, random_state=0)
    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)

    print("AdaBoostRegressor enselble Results")
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




