import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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



desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

df = pd.read_excel("Strokes.xls")
print(df.head(5))

# print(df.describe(include='all'))

df=pd.read_excel('Strokes.xls', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 87, 117, 122])
print(df.head(10))
df.to_csv('Strokefiltered.csv')


print(df.describe(include='all'))



#df.DateDeath=pd.to_datetime(df.DateDeath)
#df.DateDischarge=pd.to_datetime(df.DateDeath)
#print(df.dtypes)


# print(df['Sex'].value_counts())
# df['Sex']=df['Sex'].replace(['f'],'F')
# print(df['Sex'].value_counts())
# df['Sex']=df['Sex'].replace(['m'],'M')
# print(df['Sex'].value_counts())


#print(df['StatusDischarge'].value_counts())


#print(df['Stroke_Seq'].value_counts())
#print(df['TypeofStroke'].value_counts())
#print(df['Source'].value_counts())
#print(df['LoS'].value_counts())
#print(df['DischargeDestination'].value_counts())
#print(df['BrainScan'].value_counts())
#print(df['Followed-Up'].value_counts())



#Monthlyanalysis=df.groupby(['Sex']).agg({'BrainScan': ['count','min','max','mean', 'std', 'sum']})    # find the sum of the durations for each group
#print(Monthlyanalysis)
