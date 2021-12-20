import pandas as pd 
import ssl 
from urllib.request import urlopen
import io

import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#ignore that 
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#create dataframe
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv"
html = urlopen(url, context=ctx).read()
#print(html)
data = io.BytesIO(html)
df = pd.read_csv (data, index_col=0)
print('----------------------------')
print(df)
#print (df.head())
pd.set_option('display.max_columns', None)
# display the data types of each columns
print('----------------------------')
print(df.dtypes)

#describe the df
print('----------------------------')
print(df.describe())

#print('Number of NaN values for the column bedrooms: ', df['bedrooms'].isnull().sum())
#print('Number of NaN values for the columns bathrooms: ', df['bathrooms'].isnull().sum())

#replace NaN values by mean of column
mean = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace = True)
mean = df['bedrooms'].replace(np.nan,mean, inplace = True)
#print('Number of NaN values for the column bedrooms: ', df['bedrooms'].isnull().sum())
#print('Number of NaN values for the columns bathrooms: ', df['bathrooms'].isnull().sum())

#count number of house with unique floor values 
floor_values = df['floors'].value_counts().to_frame()
print('----------------------------')
print(floor_values)

#Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers
sns.boxplot(x="waterfront", y = 'price', data = df)
#plt.show()

#Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.
sns.regplot(x='sqft_above', y='price', data = df)
#plt.show()

#use corr() to find the feature other than price is most correlated with price

print('----------------------------')
print('correlation: ',df.corr()['price'].sort_values())

#fit a linear regression model using the longtitude feature "long" and calculate the R^2
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
print('----------------------------')
print('R^2 = ',lm.score(X, Y))

#fit a linear regression 
x = df[['sqft_living']]
y = df['price']
lml = LinearRegression().fit(x, y)
print('----------------------------')
print('R^2 = ',lml.score(x, y))

#fit a linear regression to predict 'price' using the list of features:
lm = LinearRegression()
z = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
y = df['price']
lm.fit(z,y)
print('----------------------------')
print('R^2 = ',lm.score(z,y))

#question 8: create a pipeline 
#step 1: create a tuple 
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),('model', LinearRegression())]
#step 2: 
pipe = Pipeline(Input)
pipe.fit(z,y)
print('----------------------------')
print('R^2 = ',pipe.score(z,y))

#question 9: Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.
#step 1: split data into traning and testing sets:
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
#step 2: create and fit a Ridge regression using the training data. set the regularization parameter to 0.1 and calculate the R^2 using the test data 
from sklearn.linear_model import Ridge
a = Ridge(alpha=0.1).fit(x_train,y_train)
print('----------------------------')
print('R^2 = ',a.score(x_test,y_test))

#question10: perform a polynomial transform on both the traning and testing data.
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
print('----------------------------')
print(RigeModel.score(x_test_pr, y_test))


