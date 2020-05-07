#!/usr/bin/env python
# coding: utf-8

#  *  ### In this project I am going to predict the price of used cars using Random Forest
#  *  ### Data is collected from Kaggle.
#  * ### I used python for Data cleaning, Data pre-processing, seaborn and Matplotlib for Visualisation, Scikit learn for building      the machine learning models.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Used car price prediction/craigslistVehicles.csv')


# In[ ]:


data.head()


# In[ ]:


new_data = data


# In[ ]:


new_data.shape


# ## **Data Cleaning**

# In[ ]:


#re-naming the columns
new_data.rename({'make': 'model', 'odometer': 'mileage'}, axis= 1,inplace= True)


# In[ ]:


# Lets drop the columns that are not useful for the car price prediction.
drop_columns = ['city_url', 'url','lat','long','image_url', 'VIN', 'city']
new_data = new_data.drop(drop_columns, axis =1)
new_data.shape


# In[ ]:


#lets drop the columns where price of the car is more than 2.5 million dollars because high end used cars like ferrari, porsche costs less than that
new_data = new_data.drop(new_data[new_data.price > 250000].index)
new_data.shape


# In[ ]:


#Lets drop the columns where year is less than 1990 and greater than 2019
new_data = new_data[(new_data['year'] > 1990) & (new_data['year'] < 2020)]
new_data.index = range(len(new_data))
new_data.shape


# In[ ]:


# finding the length of description of each car
new_data.dropna(subset = ['desc'],inplace = True) # dropping the null values in 'desc' column
new_data['word_len'] = new_data.desc.apply(lambda x: len(str(x.lower().split())))
new_data.index = range(len(new_data))
del new_data['desc']


# In[ ]:


#finding the age of car
from datetime import datetime

current_year = datetime.now().year
new_data['Age'] = current_year - new_data['year']
del new_data['year']


# In[ ]:


# removing the 'cylinders' in the columns and converting into float.
new_data.cylinders = new_data.cylinders.apply(lambda x: x if str(x).lower()[-1] == 'o' or str(x).lower()[-1] == 'n' else str(x).lower().replace('cylinders', ''))
new_data.cylinders = pd.to_numeric(new_data.cylinders, errors = 'coerce')
new_data.cylinders.fillna(new_data.cylinders.median(), inplace = True)


# In[ ]:


# Lets drop the rows where has more than 1 NaN values.
new_data.dropna(thresh = 13, axis = 0, inplace = True)
new_data.shape


# In[ ]:


#dropping the duplicates in the data
new_data.drop_duplicates(keep = 'first', inplace = True)
new_data.index = range(len(new_data))
new_data.shape


# In[ ]:


# fixing the spelling errors
new_data.manufacturer = new_data.manufacturer.apply(lambda x: x.replace('porche', 'porsche') if x == 'porche' else x)


# In[ ]:


# creating new dataframe to find the size of each car in order to fill the null values in the size feature of same car
new = new_data.sort_values(by= ['size','manufacturer','type'])
new.drop_duplicates(subset = ['model','type'], keep = 'first', inplace = True)
new.index = range(len(new))


# In[ ]:


new.dropna(subset = ['size','model'], inplace= True)
new.index = range(len(new))


# In[ ]:


new


# In[ ]:


# dictionary that contain size of each car
size = dict(zip(new['model'], new['size']))
for i in range(len(new_data)):
    if str(new_data['size'][i]).lower()[0] == 'n' and (new_data['model'][i] in size.keys()):
        new_data['size'][i] = size[new_data['model'][i]]


# In[ ]:


new_data


# In[ ]:


# Adding the manufacturers name to list from the dataset and adding few more manufacturer after looking the make column in dataset
manuf = []
for i in new_data.manufacturer.value_counts().index:
    manuf.append(i)
manuf.append('Tesla')
manuf.append('Rolls-Royce')
manuf.append('genesis')


# In[ ]:


# Replace the nan values in the manufacturer column based on the make column 
# from make column we can identify manufacturer of the car as it sometimes contains the manufacturer name in it
d = new_data['manufacturer']
m = new_data['model']
for i in range(len(new_data)):
    if str(d[i]).lower()[0] == 'n':
        for x in str(m[i]).lower().split():
            for mm in manuf:
                if (len(x) > 4) & (x[:4] == mm.lower()[:4]):
                    new_data['manufacturer'][i] = mm
                elif (x[:3] == mm.lower()[:3]) & (x[:-1] == mm.lower()[:-1]):
                    new_data['manufacturer'][i] = mm


# In[ ]:


#dropping the null values
new_data.dropna(subset = ['transmission','type','manufacturer','model','fuel','mileage','title_status','paint_color','drive','size','condition'],inplace = True)
new_data.index = range(len(new_data))
new_data.shape


# In[ ]:


new_data.isnull().sum()


# In[ ]:


# Function to remove the outliers in the data
def outlier_removal(x):
    for i in x:
        z_score_od = np.abs(stats.zscore(new_data[i]))
        outliers = np.where(z_score_od > 3)
        new_data.drop(new_data.index[[i for i in outliers[0]]],inplace= True)
        new_data.index = range(len(new_data))  
        
outlier_removal(new_data._get_numeric_data().columns)


# In[ ]:


#price of car cant be zero so we are eliminating the price less than 30 dollars. Some people mention very less price and they will revealthe price directly to customer while buying and bargain later.
new_data = new_data[new_data['price'] > 50]
new_data.index = range(len(new_data))
new_data.shape


# In[ ]:


#shifting the 'age' to new place
new_data.insert(1, 'age', new_data['Age'])
del new_data['Age']


# In[ ]:


# creating the dictionary for integer labelling
cond_dict = {'new':10, 'like new':9, 'excellent':8, 'good':7, 'fair':5, 'salvage':3}
title_dict = {'clean': 6, 'lien': 4, 'rebuilt':3, 'salvage': 2, 'parts only': 1, 'missing': 0}
columns = list(['condition', 'title_status'])
dictionary = list([cond_dict, title_dict])


# In[ ]:


#Function that does the integer labelling.
def labelling(columns, dictionary):
    for i in range(len(columns)):
        new_data[columns[i]] = new_data[columns[i]].map(dictionary[i])

labelling(columns, dictionary)


# In[ ]:


# Finding the models of car which has fewer data
other_models = new_data.model.value_counts().index[new_data.model.value_counts().values < 5]


# In[ ]:


# Function replaces the less frequent models of car to other_models category
def model_edit(model_list, data):
    for i in range(len(data)):
        if data[i] in model_list:
            data[i] = 'other_models'

model_edit(other_models, new_data['model'])


# In[ ]:


new_data['age'] = new_data['age'].astype(int)


# In[ ]:


new_data.head()


# In[ ]:


new_data.dtypes


# In[ ]:


new_data.isnull().sum() # Now data is clean and it has no null values


# In[ ]:


new_data.describe()


# In[ ]:


#saving the cleaned data to new csv file
new_data.to_csv('Final_data3', index= False)


# ## **Visualization**

# In[ ]:


Final_data = pd.read_csv('Final_data3')


# In[ ]:


Final_data.columns


# In[ ]:


plt.figure(figsize = (12,5))
sns.countplot(x = 'type', order = Final_data['type'].value_counts().index, data = Final_data)


# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x = 'paint_color', order = Final_data['paint_color'].value_counts().index,data = Final_data)


# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x = 'manufacturer', order = Final_data['manufacturer'].value_counts().index, data = Final_data)
plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x = 'age', order = Final_data['age'].value_counts().index, data = Final_data)


# In[11]:


Final_data.boxplot('price')


# In[10]:


sns.distplot(Final_data['price'])


# In[ ]:


sns.distplot(Final_data['mileage'])


# In[12]:


sns.distplot(Final_data['word_len'])


# In[13]:


Final_data.boxplot('mileage')


# In[14]:


Final_data.boxplot('word_len')


# In[17]:


plt.figure(figsize = (10,5))
corr = Final_data.corr()
sns.heatmap(corr, annot = True)


# In[18]:


sns.pairplot(Final_data)


# ## **Model building**

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model  import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost


# In[ ]:


Final_data = pd.read_csv('Final_data3')


# In[ ]:


Final_data.head()


# In[ ]:


F1 = pd.get_dummies(Final_data, drop_first= True)
F1.head()


# In[ ]:


F1.shape


# In[ ]:


X = F1.iloc[:, 1:]
y = F1.iloc[:,0:1]


# In[ ]:


X = np.array(X)
y = np.array(y).reshape(-1)


# In[ ]:


# scaling the inputs
scaler = MinMaxScaler().fit(X)
scaledX = scaler.transform(X)


# In[ ]:


# Algorithms
lasso = Lasso()
ridge = Ridge()
Dtree = DecisionTreeRegressor()
sgd = SGDRegressor()
gradboost = GradientBoostingRegressor()
rfreg100 = RandomForestRegressor(n_estimators = 100)
xgbreg = xgboost.XGBRegressor()


# In[ ]:


# parameters
kfold = KFold(n_splits = 5)
scoring = 'r2'
algo_list = list([rfreg100])#xgbreg,rfreg1,rfreg2,lasso,ridge,Dtree,sgd,gradboost])
algo_name = list(['rfreg100'])#'xgbreg','rfreg1','rfreg2','lasso','ridge','Dtree','sgd','gradboost'])


# In[ ]:


# Buliding models on various algorithms
def model_building(algo,X,y,fold,scoring):
    algo_score = []
    for i in algo:
        score = cross_val_score(i, X, y, cv=fold, scoring=scoring).mean()
        algo_score.append(score)
    return algo_score


# In[ ]:


training = model_building(algo_list, scaledX, y, kfold, scoring)


# In[ ]:


training_score = dict(zip(algo_name,training))


# In[ ]:


training


# In[ ]:





# In[ ]:


# prediction
def prediction(algo,X,y,fold,scoring):
    algo_score = []
    for i in algo:
        pred_score = cross_val_predict(i, X, y, cv=fold, scoring=scoring).mean()
        algo_score.append(pred_score)
    return algo_score


# In[ ]:


testing = model_building(algo_list, scaledX, y, kfold, scoring)


# In[ ]:


testing_score = dict(zip(algo_name,testing))


# In[ ]:


get_ipython().system('pip install joblib')


# In[ ]:


rfreg100.fit(scaledX, y) #12:09 pm


# In[ ]:


# saving the model
import joblib

rf_model = 'rf_final_model.sav'
joblib.dump(rfreg100, rf_model)


# In[ ]:


Final_score


# In[ ]:


Final_score


# In[ ]:


Final_score # old code 10,50,100


# In[ ]:




