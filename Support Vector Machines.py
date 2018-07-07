import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print (cancer.keys())
#print (cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, cancer['target'], test_size=0.33, random_state=42)

from sklearn.svm import SVC #Support Vector Clustering
model = SVC()

# without adjusting parameters this model put all test data into one class.
#model.fit(X_train, y_train)
#predict = model.predict(X_test)

#Grid Search - finding right C & gamma values to use for parameters
from sklearn.grid_search import GridSearchCV
param_grid = {'C':[.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
#Verbose = amount of text output of description of process you want.

grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid.best_score_

predict = grid.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
