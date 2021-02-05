import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 60)

df=pd.read_csv('randomforest.csv')
print(df.head)

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

X=df.iloc[:, :-1].values
y=df.iloc[:, -1].values


# X=df.values
# y=df['loan'].values

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.8, random_state=0)

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier=RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
