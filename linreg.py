from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Training the model
def linreg(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    # Testing the model
    y_pred = model.predict(X_test)

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