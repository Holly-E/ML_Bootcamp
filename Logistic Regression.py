import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
# sns.countplot(x='Survived', hue='Sex', data=train)
# sns.countplot(x='Survived', hue='Pclass', data=train)
# sns.distplot(train['Age'].dropna(), kde=False, bins=30)
# sns.countplot(x='Parch', data=train)
# train['Fare'].hist(bins=40)
# sns.boxplot(x='Pclass', y='Age', data=train) #view avg age by class, could also call mean on the row

def impute_age(cols):
    #return/fill in missing age by class
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24

    else:
        return Age


# Sweet way to perform function on columns!
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis=1)
#axis = 1 to apply across the columns

train.drop('Cabin', inplace=True, axis=1)
train.dropna(inplace=True)
test.drop('Cabin', inplace=True, axis=1)
test.dropna(inplace=True)
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis') #now age is populated!

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
pclass = pd.get_dummies(train['Pclass'], drop_first=True)

sex1 = pd.get_dummies(test['Sex'], drop_first=True)
embark1 = pd.get_dummies(test['Embarked'], drop_first = True)
pclass1 = pd.get_dummies(test['Pclass'], drop_first=True)


train = pd.concat([train,sex,embark,pclass], axis=1)
train.drop(['Name', 'Sex', 'Ticket', 'Embarked', 'PassengerId', 'Pclass'], axis=1, inplace=True)

test = pd.concat([test,sex1, embark1,pclass1], axis=1)
test.drop(['Name', 'Sex', 'Ticket', 'Embarked', 'PassengerId', 'Pclass'], axis=1, inplace=True)

X_train = train.drop('Survived', axis=1) #cool way to not list all other columns
y_train = train['Survived']

X_test = test #cool way to not list all other columns
#y_test = test['Survived']

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

#Instantiate an instance object (model)
lm = LogisticRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

from sklearn.metrics import classification_report
#print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test, predictions))

# Not sure if the following coefficients are relevant for logistical, instead convert to Odds Ratios??
# print(pd.DataFrame(lm.coef_[0], X_train.columns,columns=['Coeff']))

plt.tight_layout() # So labels don't get cutoff
plt.show()