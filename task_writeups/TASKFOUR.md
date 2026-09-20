# Task Four: What's Next?

[Back to README](../README.md)

A new use case for the optimized linear regression, which in the case of task two was the ridge regression regularization method, would be to accurately predict the number of Cycle Count tasks that will be generated for warehouse associates to complete.

For context, system generated inventory tasks will populate at the end of day each day. These are generated at 4:30 pm and the number of tasks is not known until the script is ran. As an operator, it is adamant that the number of tasks is known beforehand for planning purposes. Thus, the use of the ridge regression will help predict the number of tasks generated based on a collection of data.

First, the domain’s scope of importance is in its data. Features that are necessary are units shipped out, shortages during selection, units received, produce received, the day of the week, next day projections, etc. The constraints however *is* the correlation of each day with another. Because products naturally have a shelf life greater than one day, inventory tasks are not only dependent on the features for one specific day (or row in the table), but also the previous days beforehand. For this case, a feature engineering method used will be autoregression, formatting past time steps, i.e. lags, and adding them together: [^1]

$$y_{t} = c + \phi_{1}y_{t - 1} + \phi_{2}y_{t - 2} + \ldots + \phi_{p}y_{t - p} + \varepsilon_{t}$$

Given the average shelf life and units on hand are four days of outbound product, the lag will consist of the four previous operational days. This will allow the model to take the total four days of data into consideration to generate a more accurate prediction. The use of a ridge regression and this feature engineering is beneficial due to the multicollinearity. A ridge regression model is designed to specifically manage such a case by penalizing large coefficients. Thus, the autoregression engineering and ridge regression collaborate to create an optimal prediction.

To address the elephant in the room, why not just run the script at the beginning of the day and get the number of tasks that would generate? This will simply not work due to the nature of how much product would be selected, what shortages would happen, and multiple other factors. Thus, using this ridge regression model to determine the number of tasks generated is a better use case to assist with operational planning.

Because simple addition is used in the feature engineering, the performance and scalability of this model is similar to the previous use case. There is no theoretical limit to how much data can be entered into the model for training, meaning the aforementioned metrics are simply constrained by compute power and time. Performance in terms of accuracy, however, will be measured the same as in the previous use case. The R2 score and the Mean Squared Average will provide the accuracy of the model by comparing the model’s daily predictions and the actual system generated tasks at 4:30 pm.

Integration will be a simple application based on the previous operational day’s data that is entered by the operator. Post training, the weighted model will be uploaded to the application and with first use, the operator will input the previous four days. Afterwards, the application will store the previous four days of data naturally so that the operator will simply input the previous day and not the three preceding. Finally, potential regulations that may affect the use of this model should simply align with the operational security policies that are in place. I.e., data should not be shared outside of the facility.

[Back to README](../README.md)

[^1]: Pennsylvania State University, "10.2 - Autocorrelation and Time Series Methods," *STAT 462: Applied Regression Analysis*, accessed August 14, 2026, <https://online.stat.psu.edu/stat462/node/188/>.