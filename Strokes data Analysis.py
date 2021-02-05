import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.stats import *
import seaborn as sns
import seaborn as sb


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 20)


#---Read in data set
df = pd.read_csv("Strokesdataset.csv")                                                 #reads CSV file
print(df.head(5))                                                                       #prints top 5 values

#---Summary of data
print(df.describe(include='all'))


#--cleaning up Sex column
print(df['Sex'].value_counts())
df['Sex']=df['Sex'].replace(['f','F'],'Female')                                         #replace f and F with Female
df['Sex']=df['Sex'].replace(['m','M'],'Male')
print(df['Sex'].value_counts())

G1=df.hist(column='Age')
plt.show()
print('STD')
print(df.loc[:,'Age'].std())                                                            #standard deviation of age
print('Mean')
print(df['Age'].mean())
print('Median')
print(df['Age'].median())

##mean of Age
Agemeansex=(df.groupby('Sex')['Age'].mean())                                            #gives average of Age of male and female
print('Mean age of Male and Female')
print(Agemeansex)

print(df['Sex'].value_counts(normalize=True) * 100)                                     #gives a percentage value



# #---Cleaning up TypeofStroke column
#
print(df['TypeofStroke'].value_counts())                  #gives count of different Stokes



df['TypeofStroke']=df['TypeofStroke'].replace(['INFARCT','Infarct'],'Infarction')
df['TypeofStroke']=df['TypeofStroke'].replace(['BLEED'],'Bleed')
df['TypeofStroke']=df['TypeofStroke'].replace(['SAH'],'Subarachnoid')


print(df['TypeofStroke'].value_counts())

G2=df['TypeofStroke'].value_counts().plot(kind='pie',autopct='%0.1f%%', figsize=(6,6))
plt.show(G2)
print(df['TypeofStroke'].value_counts(normalize=True) * 100)


# #---status discharge
print(df['StatusDischarge'].value_counts())
print(df['StatusDischarge'].value_counts(normalize=True) * 100)
plt.show()
G3=df['StatusDischarge'].value_counts().plot(kind='pie',autopct='%0.1f%%', figsize=(6,6))
plt.show(G3)


Strokedeath1=df.groupby(['TypeofStroke'])['StatusDischarge'].value_counts()
print(Strokedeath1)
Strokedeath2=df.groupby(['TypeofStroke'])['StatusDischarge'].value_counts(normalize=True) * 100
print(Strokedeath2)


 #------length of stay
G4=df.hist(column='LoS')
plt.show(G4)
print(df['LoS'].value_counts())                                                                         #gives count of Los
print(df['LoS'].mean())
df=df.round({'LoS':0})                                                                      #rounds value up to the nearest whole number
print(df['LoS'].value_counts())
print(df['LoS'].mean())
# df.to_csv('StrokesT2.csv')
LosbyStroke=(df.groupby('TypeofStroke')['LoS'].mean())
print(LosbyStroke)

# #-----time series analysis
df.DateAdmission=pd.to_datetime(df.DateAdmission)
df['Weekdays']=df.DateAdmission.dt.weekday_name                                      #adds column of what weekday according to date column
df['Month']=df.DateAdmission.dt.month                                               #adds column of what Month according to date column
df['Week']=df.DateAdmission.dt.week                                                  #adds column of what week number according to date column
df['Year']=df.DateAdmission.dt.year

print(type(df.DateAdmission))
# df['Hour']=df.DateAdmission.dt.hour                                                                                               #adds hour column
# print(df.head())


print(df.head)

monthgraph=df.groupby('Month').Month.count().plot(kind='line', figsize=(15,4))
plt.show(monthgraph)

weekgraph=df.groupby('Week').Week.count().plot(kind='line', figsize=(15,4))
plt.show(weekgraph)

yeargraph=df.groupby('Year').Week.count().plot(kind='line', figsize=(15,4))


#------Brain Scan Analysis
                                                                       #replace blanks with No

print(df['BrainScan'].value_counts())                                                                         #gives count of Los

BrainScandeath=df.groupby(['BrainScan'])['StatusDischarge'].value_counts()
print(BrainScandeath)

BrainScandeath1=df.groupby(['BrainScan'])['StatusDischarge'].value_counts(normalize=True) * 100
print(BrainScandeath1)

#----Cholesterol_AD

Cholesterol_AD_hit=df.hist(column='Cholesterol_AD')
plt.show(Cholesterol_AD_hit)

HDL_AD_hit=df.hist(column='HDL_AD')
plt.show(HDL_AD_hit)


LDL_AD_hit=df.hist(column='LDL_AD')
plt.show(LDL_AD_hit)

print(df.describe(include='all'))


sexvsCholesterol_ADmean=(df.groupby('Sex')['Cholesterol_AD'].mean())                                            #gives average of Age of male and female
print('Mean age of Male and Female')
print(sexvsCholesterol_ADmean)

sexvsHDL_ADmean=(df.groupby('Sex')['HDL_AD'].mean())                                            #gives average of Age of male and female
print(sexvsHDL_ADmean)

sexvsHDL_ADmean=(df.groupby('Sex')['LDL_AD'].mean())                                            #gives average of Age of male and female
print(sexvsHDL_ADmean)

# G2=df['Cholesterol_AD'].value_counts().plot(kind='bar')
# plt.show(G2)

# print(Strokedeath2)
# Cholesterol=df.groupby(['TypeofStroke','StatusDischarge'])['Cholesterol_AD'].mean()
# print(Cholesterol)
# Cholesterol1=df.groupby(['TypeofStroke','StatusDischarge'])['HDL_AD'].mean()
# print(Cholesterol1)
# Cholesterol2=df.groupby(['TypeofStroke','StatusDischarge'])['LDL_AD'].mean()
# print(Cholesterol2)
#
#

# corr=df.corr()
# print(corr)
# # print(df.corr(method='pearson'))
#
#
#
df.dropna(inplace=True)


print(df.corr(method='pearson'))


print(df.describe(include='all'))
#
heatmap=sns.heatmap(df.corr())
plt.show(heatmap)


# df.to_csv('Strokes1.csv')

# pairplot=sb.pairplot(df)
# plt.show(pairplot)



# corr=df.corr()
# print(corr)
# #
# df['Age'].apply(np.ceil)
# df['LoS'].apply(np.ceil)
# print(df['Age'].value_counts())
# print(df['LoS'].value_counts())
# #
# #
agevslosscatter=plt.scatter(df.Age,df.LoS, color='red', marker='+')
plt.show(agevslosscatter)

slope,intercept,r_value,p_value,std_err = linregress(df.Age, df.LoS)
print('R Squared Value')
print((pow(r_value,2)))
print('P-Value')
print(p_value)
print('Gradiant')
print(slope)
print('Y-Intercept')
print(intercept)
#
regression = linear_model.LinearRegression()
predictagelos=regression.fit(df[['Age']],df.LoS)
#
print(predictagelos.predict([[50]]))

df2 = pd.read_csv("strokesage.csv")
print(df2.head())
df2['strokesage']=predictagelos.predict(df2)
print(df2)

df2.to_csv('Strokesagelos.csv')


import statsmodels.api as sm

X=df["Age"]
y=df["LoS"]

model=sm.OLS(y,X).fit()
prediction=model.predict(X)
print(model.summary())



agevslosreg = linear_model.LinearRegression()
agevslosreg.fit(df[['Age']],df.LoS)
agevslosreggraph=sns.regplot(x='Age', y='LoS', data=df);
plt.show(agevslosreggraph)
#
#
agevsCholesterol_AD=plt.scatter(df.Age,df.Cholesterol_AD, color='red', marker='+')
plt.show(agevsCholesterol_AD)
# #
slope,intercept,r_value,p_value,std_err = linregress(df.Age,df.Cholesterol_AD)
print('R Squared Value')
print((pow(r_value,2)))
print('P-Value')
print(p_value)
print('Gradiant')
print(slope)
print('Y-Intercept')
print(intercept)
#
regression = linear_model.LinearRegression()
regression.fit(df[['Age']],df.Cholesterol_AD)
agevsCholesterol_ADgraph=sns.regplot(x='Age', y='Cholesterol_AD', data=df);
plt.show(agevsCholesterol_ADgraph)

# #
# #
slope,intercept,r_value,p_value,std_err = linregress(df.Age,df.HDL_AD)
print('R Squared Value')
print((pow(r_value,2)))
print('P-Value')
print(p_value)
print('Gradiant')
print(slope)
print('Y-Intercept')
print(intercept)
# #
regression = linear_model.LinearRegression()
regression.fit(df[['Age']],df.HDL_AD)
agevsHDL_ADgraph=sns.regplot(x='Age', y='HDL_AD', data=df);
plt.show(agevsHDL_ADgraph)

#
#
slope,intercept,r_value,p_value,std_err = linregress(df.Age,df.LDL_AD)
print('R Squared Value')
print((pow(r_value,2)))
print('P-Value')
print(p_value)
print('Gradiant')
print(slope)
print('Y-Intercept')
print(intercept)
# #
regression = linear_model.LinearRegression()
regression.fit(df[['Age']],df.LDL_AD)
agevsLDL_ADgraph=sns.regplot(x='Age', y='LDL_AD', data=df);
plt.show(agevsLDL_ADgraph)
#




