#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

import os 


df_auto = pd.read_csv('/Users/anuragshrestha/Desktop/Multi-Linear-/Car Prediction Project/CarPrice_Assignment.csv')

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

#### changing the carname errors #### 

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
print("the count of Outliers Price is", OutliersPrice.count())


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

#sns.pairplot(coloumns_numbers, vars = ['price',
#'wheelbase', 'carlength', 'carwidth','carheight',        
#'curbweight',      
#enginesize'])

# engine size, curweight, car width, wheel base and car length has a linear 

#sns.pairplot(coloumns_numbers, vars = ['price', 
#'boreratio',        
#'stroke',              
#'compressionratio',   
#'horsepower',       
#'peakrpm',        
#'citympg',             
#'highwaympg'])

#horsepower and boreratio has postive linear correlation with the price 
# city and Highway mpg have negative correlation with price 


#looking at categorical types 

cateorical_features = df_new.select_dtypes(exclude = ['int', 'float'])


for feature in cateorical_features:  
    plt.figure(figsize= (10,10))
    sns.boxenplot(x = feature, y = 'price', data = df_new)


# the door number seems to have an effect on the prices 
#carbody varies in all the prices -- with common knowledge, the
# car body will have an impact on the prices 
# cylindernumber = the amount of horse power. 
#drive wheel should have an impact on the prices as well 

#distrubtion of numberical variables 

numberical_variables = [feature for feature in df_new if feature not in cateorical_features]

for feature in numberical_variables:
    data=df_new.copy()
    data[feature].hist(bins=15)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


#Selecting the important features  

important_var = ['price', 'enginesize', 'curbweight','carwidth', 
    'wheelbase','carlength', 'horsepower', 'boreratio',
    'citympg','highwaympg', 'doornumber','carbody',
    'cylindernumber', 'drivewheel', 'symboling', 'aspiration']
    
df_new = df_new[important_var]
    
important_cat= ['doornumber','carbody', 'cylindernumber', 
    'drivewheel', 'symboling', 'aspiration']

dummies_cat = pd.get_dummies(df_new[important_cat], drop_first= True)

df_new = pd.concat([df_new, dummies_cat], axis= 1) 
#connecting the dummy variables to the data frame 

df_new.drop(important_cat, axis = 1, inplace = True)



df_new.to_csv('/Users/anuragshrestha/Desktop/Multi-Linear-/Car Prediction Project/train.csv')



# %%
