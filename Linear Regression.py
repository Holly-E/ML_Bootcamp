import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Data/USA_HOUSING copy.csv')

"""
Within the SciView, click on glasses icon to view special variables
Click "View as DataFrame" for full view, or df.head() or df.corr().
The data will be presented in the Data tab next to plots
"""
#print (df.head())
#print (df.info())
#print (df.describe())
#print (df.columns) - Use to grab column names for training model

#sns.pairplot(df)
#sns.distplot(df['Price'])
#sns.heatmap(df.corr(), annot=True)

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

#Instantiate an instance object (model)
lm = LinearRegression()
lm.fit(X_train, y_train)

#Evaluate model
print(lm.intercept_)
print(lm.coef_)

cdf = pd.DataFrame(lm.coef_, X.columns,columns=['Coeff'])

predictions = lm.predict(X_test)
#plt.scatter(y_test, predictions)

sns.distplot((y_test - predictions)) # histogram of residuals

from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
r_value = metrics.explained_variance_score(y_test, predictions)

plt.tight_layout() # So labels don't get cutoff
plt.show()