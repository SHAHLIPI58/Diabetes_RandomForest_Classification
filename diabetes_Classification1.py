# -*- coding: utf-8 -*-
"""
Created on Thu May 31 07:45:08 2018

@author: ll
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pima = pd.read_csv("diabetes.csv")
pima.info()
pima.describe()
pima.shape
pima.head(20)

# column names as a list
col = pima.columns      
print(col)

#observation : many data are filled which 0--> which is not applicable
dataset = pd.read_csv('diabetes.csv')
print((dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']] == 0).sum())
print(col)

# mark zero values as missing or NaN

dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']]= dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']].replace(0, np.NaN)
# count the number of NaN values in each column
print(dataset.isnull().sum())

dataset.shape
dataset.info()



# fill missing values with mean column values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='0', strategy='mean', axis=0)

print(dataset.isnull().sum())
dataset_x1 =dataset.apply(lambda dataset: dataset.fillna(dataset.mean()),axis=0)
print(dataset_x1.isnull().sum())



pima = dataset_x1
#find missing value
pima.isnull().sum()
pima.info()

y = pima.Outcome
list = ['Outcome']
x = pima.drop(list,axis = 1 )
#x = pima.columns[0:len(pima.columns) - 1]  (above two line of code into single line)
x.head()

#count Outcome 0 and 1
pima.groupby("Outcome").size()
ax = sns.countplot(y,label="Count") 

#visualization of features: Histrogram is useful when some outliner value entered in db
#at that time mean, meadian feature sometimes mislead us
pima.hist(figsize=(12,8))

#box plot to visualise the range of data available with meadian of each feature and outlier by rounds
pima.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))

#check which features are similar or almost similar by violin plot and swarm plot
pima.describe()
#before that data shoes be standarized to plot
x_normalise =(x - x.mean()) / (x.std())  

data_con= pd.concat([y,x_normalise.iloc[:,0:8]],axis=1)
data = pd.melt(data_con,id_vars="Outcome",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="Outcome", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

#observe: all features are different from each other except age and DiabetesPedigree Function
#make correlation plot between them
sns.jointplot(x.loc[:,'DiabetesPedigreeFunction'], x.loc[:,'Age'], kind="regg", color="#ce1414")
# our assumption is wrong as we get p = 0.034 --> which is very low

              
#make swan plot to find which feature is useful for classification
sns.set(style="whitegrid", palette="muted")
data_dia = y
datax = x
data_std = (datax - datax.mean()) / (datax.std())              # standardization


data = pd.concat([y,data_std.iloc[:,0:8]],axis=1)
data = pd.melt(data,id_vars="Outcome",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="Outcome", data=data)
plt.xticks(rotation=90)



#observation : glucoss ,BMI  and insulin play role to classify

#now apply different feature selection techniques to find out required actual features
#we will find max 4-5 features which has high weightage

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# find best scored 4 features
select_feature = SelectKBest(chi2, k=4).fit(x, y)

print('Score list:', select_feature.scores_)
print('Feature list:', x.columns)
x = select_feature.transform(x)


#now check correlation between this features by heatmap
# drop useless features

list = ['Pregnancies','BloodPressure','DiabetesPedigreeFunction','Outcome','SkinThickness']
x_required = pima.drop(list,axis = 1 )
x_required.head()

f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(x_required.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


#apply classification models

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x_required, y, test_size=0.3, random_state=42)



clf_rf = RandomForestClassifier(criterion= 'entropy', min_samples_leaf= 6, min_samples_split= 2, n_estimators= 16)      
clr_rf = clf_rf.fit(x_train,y_train) 
y_pred = clf_rf.predict(x_test)
#-->73.59307359307359 % accuracy
ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



#with k fold tuning and gridsearch


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clr_rf, X = x_train, y = y_train, cv = 10)
accuracies_mean = accuracies.mean()
accuracies.std()


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestClassifier

#hyper parameters set
params = {'criterion':['gini','entropy'],
          'n_estimators':[15,16,17,18,19],
          'min_samples_leaf':[2,3,4,5,6,7,8],
          'min_samples_split':[2,3,4], 
         
          
         }
#Making models with hyper parameters sets
model1 = GridSearchCV(clr_rf, param_grid=params, scoring = 'accuracy')
#learning
model1.fit(x_train,y_train)
#The best hyper parameters score and set
best_accuracy = model1.best_score_
print("Best Hyper Parameters:\n",model1.best_params_)











