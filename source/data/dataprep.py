from sklearn.model_selection import train_test_split

import pandas as pd

# Importing the dataset using Pandas
data = pd.read_csv('Dataset.csv')

# Here is where we are splitting up the dataset into features and a target.
features = data.iloc[0:1000, 0:-1]
target = data.iloc[0:1000, -1]

# Here we are taking the split dataset and creating our training and testing set.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)