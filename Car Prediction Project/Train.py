#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
import os 


# machine learning 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn  import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn import metrics 
from sklearn import pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


df_train = pd.read_csv('/Users/anuragshrestha/Desktop/Multi-Linear-/Car Prediction Project/train.csv'
, index_col= 0)


# transforming the price by 1000's 

df_train['price'] = df_train['price'].div(1000)



#scaling the datas 

feature_scaling = [feature for feature in df_train.columns if feature not in ['price']]

scaler = MinMaxScaler()
scaler.fit(df_train[feature_scaling])
print(scaler.transform(df_train[feature_scaling]))

#connected the price and the scalted datas 
data_train = pd.concat([df_train[['price']].reset_index(drop=True),
 pd.DataFrame(scaler.transform(df_train[feature_scaling]), columns=feature_scaling)],
 axis=1)

print(data_train)


#feature selection 
X_train = data_train.drop(['price'], axis= 1)
y_train = data_train['price']

# RFE method to select features 

estimator = linear_model.LinearRegression()

features = RFE(estimator)

features.fit(X_train, y_train)

selected_features = X_train.columns[(features.get_support())]

print(selected_features)

X_train = X_train[selected_features]

## splitting the data set ##
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

## linear regression 
LR = linear_model.LinearRegression()
LR.fit(X_train,y_train)

y_prediction = LR.predict(X_test)

#evaluation of the model 

print("the r2 score of the train model is:", r2_score(y_test, y_prediction))
print("the accuracy of the test model is:", LR.score(X_train, y_train))

print(" the MAE of the test data set:", mean_absolute_error(y_test, LR.predict(X_test)))
print(" the MAE od the train data set:", mean_absolute_error(y_test, y_prediction))


plt.figure(figsize=(15,10))

plt.scatter(y_test, y_prediction, c='green')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', c='red', lw=3)
plt.xlabel('Actuals')
plt.ylabel('Predicted Values')
plt.title('Actuals Vs Predicted Values')


# Ploting Residuals

plt.figure(figsize=(15,10))

sns.residplot(y_test, y_prediction, color='green')
plt.xlabel('Actual Price')
plt.ylabel('Residuals')
plt.title('Actuals Vs Residuals')


#model's coeffients 
coefficient = LR.coef_
print(coefficient.shape)

coeff = coefficient.reshape(-1,11)

df_model = pd.DataFrame(coeff, columns= [X_train.columns])

print(df_model)

df_model.to_csv('/Users/anuragshrestha/Desktop/Multi-Linear-/Car Prediction Project/df_model.csv')
#%%