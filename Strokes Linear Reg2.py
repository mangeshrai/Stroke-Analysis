import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import tensorflow
# import keras
import seaborn as sns
from pylab import rcParams
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import scale
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xlrd
from scipy.stats import *
from sklearn.utils import shuffle
from scipy.stats.stats import pearsonr


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 20)

df = pd.read_csv("Strokesfiltered.csv")
print(df.head(5))

print(df.describe(include='all'))                                                       #goves overal statistical view of data

df.DateDeath=pd.to_datetime(df.DateDeath)
df.DateDischarge=pd.to_datetime(df.DateDeath)                                           #changes date into date time mode
print(df.dtypes)

print(df['Sex'].value_counts())                                                         #counts males and females
df['Sex']=df['Sex'].replace(['f','F'],'Female')
df['Sex']=df['Sex'].replace(['m','M'],'Male')
print(df['Sex'].value_counts())

print(df['TypeofStroke'].value_counts())                                                    #gives count of different Stokes
df['TypeofStroke']=df['TypeofStroke'].replace(['INFARCT','Infarct'],'Infarction')
df['TypeofStroke']=df['TypeofStroke'].replace(['BLEED'],'Bleed')
df['TypeofStroke']=df['TypeofStroke'].replace(['SAH'],'Subarachnoid')
print(df['TypeofStroke'].value_counts())
plt.show(df['TypeofStroke'].value_counts().plot(kind='pie',autopct='%0.1f%%', figsize=(6,6)))
print(df['TypeofStroke'].value_counts(normalize=True) * 100)
df['Age'].apply(np.ceil)
df['LoS'].apply(np.ceil)

df.dropna(inplace=True)
corr=df.corr()
print(corr)

plt.scatter(df.Age,df.LoS)
plt.xlabel('Age')
plt.ylabel('LoS')
plt.show()

reg=linear_model.LinearRegression()
reg.fit(df[['Age']],df.LoS)


print(reg.predict([[70]]))

print('Gradiant')
print(reg.coef_)
print('Intercept')
print(reg.intercept_)

plt.scatter(df.Age,df.LoS)
plt.xlabel('Age')
plt.ylabel('LoS')
plt.scatter(df.Age,df.LoS,color='red')
plt.plot(df.Age,reg.predict(df[['Age']]),color='blue')

plt.show()

