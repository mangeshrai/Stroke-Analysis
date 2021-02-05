import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import *
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sb

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 25)
pd.set_option("display.max_rows", 1000)

df=pd.read_csv('Strokesdataset.csv')
print(df.head(5))

df=pd.read_csv('Strokesdataset.csv', usecols=[1,2,4,6,7,8,9,10,11])
print(df.head(5))


print(df.describe(include='all'))
print(df.head(5))

df.dropna(inplace=True)
print(df.describe(include='all'))

# #--cleaning up Sex column
print(df['Sex'].value_counts())
df['Sex']=df['Sex'].replace(['f','F'],'Female')                                         #replace f and F with Female
df['Sex']=df['Sex'].replace(['m','M'],'Male')
print(df['Sex'].value_counts())
print(df.describe(include='all'))
#
#
#
df['StatusDischarge'].replace(1,'0', inplace=True)
df['StatusDischarge'].replace(2,'1', inplace=True)
print(df['StatusDischarge'].value_counts())
print(df.head(5))
print(df.describe(include='all'))

print(df['TypeofStroke'].value_counts())                                                    #gives count of different Stokes
df['TypeofStroke']=df['TypeofStroke'].replace(['INFARCT','Infarct'],'Infarction')
df['TypeofStroke']=df['TypeofStroke'].replace(['BLEED'],'Bleed')
df['TypeofStroke']=df['TypeofStroke'].replace(['SAH'],'Subarachnoid')
print(df['TypeofStroke'].value_counts())

df['Sex'] = pd.factorize(df.Sex)[0]
df['TypeofStroke'] = pd.factorize(df.TypeofStroke)[0]
df['BrainScan'] = pd.factorize(df.BrainScan)[0]

print(df.head(5))
print(df.describe(include='all'))

X=df.drop('StatusDischarge', axis=1)
y=df['StatusDischarge']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.5, random_state=1)
# # print(X_train.shape)
log_reg = LogisticRegression(solver='lbfgs', max_iter=300)
log_reg.fit(X_train, y_train)


y_pred=log_reg.predict(X_test)
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

df2 = pd.read_csv("LogisticregStrokes.csv")
print(df2.head())
df2['LogisticregStrokes']=log_reg.predict(df2)
print(df2)

# #
# # plt.scatter(df.Age,df.StatusDischarge, marker='+', color='red')
# # plt.show()
# #
# # print(df.shape)
# #
# # X_train, X_test, y_train, y_test = train_test_split(df[['Age']],df.StatusDischarge, random_state=1)
# # print(X_train.shape)
# #
# # log_reg = LogisticRegression(solver='lbfgs')
# # log_reg.fit(X_train, y_train)
# #
# # y_pred=log_reg.predict(X_test)
# #
# # print(confusion_matrix(y_test, y_pred))
# # print(log_reg.predict(X_test))
# # print(log_reg.score(X_test, y_test))
# # print(log_reg.predict_proba(X_test))
# #
# # df2 = pd.read_csv("LogisticregStrokes.csv")
# # print(df2.head())
# # df2['LogisticregStrokes']=log_reg.predict(df2)
# # print(df2)