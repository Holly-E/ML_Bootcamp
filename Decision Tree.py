import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/kyphosis.csv')

#sns.pairplot(data = df, hue='Kyphosis')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Kyphosis', axis=1), df['Kyphosis'], test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predict = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predict))
print(confusion_matrix(y_test, predict))

#Use Random Forest to get a better result!!
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)

print(classification_report(y_test, rfc_predict))
print(confusion_matrix(y_test, rfc_predict))
plt.show()
