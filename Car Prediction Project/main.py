#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from datetime import datetime, timedelta


# machine learning 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

# graphs 
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)
sns.set(style='darkgrid')
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker

df_auto = pd.read_csv('CarPrice_Assignment.csv')

print(df_auto.head) 

print(df_auto.columns)

print(df_auto.describe())

print(df_auto.dtypes)

#data cleaning 

print(df_auto.info())
# all the columns have 205 values so there are not any missing values 

df_new = df_auto.drop(columns= 'car_ID')
print(df_new)

CarNames = df_auto.CarName.str.split(' ', expand = True)
print(CarNames)

df_new['CarName'] = CarNames

#changing the carname errors 
df_new['CarName'] = df_new['CarName'].replace({'maxda': 'mazda', 'Nissan': 'nissan', 
'porcshce': 'porsche', 'toyouta':'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'})

print(df_new['CarName'].unique()) #replaced the name errors

# analyzing correlations and outliers of price and other variables 

sns.boxplot(df_new['price'])

# there are multiple outliers after the price of 30,000
#analyzing the outliers prices 

plt.figure(figsize= (15,15))
sns.barplot( x = df_new['CarName'], y = df_new['price'])



OutliersPrice = df_new[df_new.price > 30000]
print(OutliersPrice.head())
print(OutliersPrice.count())

#what does the boxplot show for the prices of the car? 
# there is an outlier for the prices that are greater than 31,000 
# the porche seems to have higher prices because of the brand 
# keep the outliars the same for now 

#visualation for the variables 

plt.figure(figsize= (15,15))
df_new['CarName'].value_counts().plot(kind = 'bar')
plt.xlabel("Cars")
plt.ylabel("Count")


plt.figure(figsize= (8,8))
sns.distplot(df_new['price'])

# correlation between price and other variables 

coloumns_numbers = df_new.select_dtypes(exclude = ['object'])
print("count of float and int:", coloumns_numbers.dtypes) 





sns.pairplot(coloumns_numbers, vars = ['price',
'wheelbase', 'carlength', 'carwidth','carheight',        
'curbweight',      
'enginesize'])

# engine size, curweight, car width, wheel base and car length has a linear 

sns.pairplot(coloumns_numbers, vars = ['price', 
'boreratio',        
'stroke',              
'compressionratio',   
'horsepower',       
'peakrpm',        
'citympg',             
'highwaympg'])

#horsepower and boreratio has postive linear correlation with the price 
# city and Highway mpg have negative correlation with price 

#looking at categorical types 

columns_cat = df_new.select_dtypes(exclude = ['int', 'float'])
print(columns_cat)

plt.figure(figsize=(20, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'doornumber', y = 'price', data = df_new)
plt.subplot(3,3,2)
sns.boxplot(x = 'fueltype', y = 'price', data = df_new)
plt.subplot(3,3,3)
sns.boxplot(x = 'aspiration', y = 'price', data = df_new)
plt.subplot(3,3,4)
sns.boxplot(x = 'carbody', y = 'price', data = df_new)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = df_new)
plt.subplot(3,3,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = df_new)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginetype', y = 'price', data = df_new)
plt.subplot(3,3,8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = df_new)
plt.subplot(3,3,9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df_new)
plt.show()

# the door number seems to have an effect on the prices 
#carbody varies in all the prices -- with common knowledge, the
# car body will have an impact on the prices 
# cylindernumber = the amount of horse power. 
#drive wheel should have an impact on the prices as well 

important_var = ['enginzesize', 'curbweight','carwidth', 
'wheelbase','carlength', 'horsepower', 'boreratio',
'citympg','highwaympg', 'doornumber','carbody',
'cylindernumber', 'drivewheel']

impotant_ cat= ['doornumber','carbody', 'cylindernumber', 
'drivewheel']

# %%
