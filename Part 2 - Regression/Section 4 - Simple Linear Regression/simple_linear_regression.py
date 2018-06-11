#simple linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
datasets = pd.read_csv('Salary_Data.csv')
x = datasets.iloc[:, :-1].values
y = datasets.iloc[:, 1].values

#splitting the dataset into training_set and test_set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)

#visualizing the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()