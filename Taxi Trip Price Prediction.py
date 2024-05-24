import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df1=pd.read_csv('C:/DATASCIENCE/MyProjects/My_Projects/DATASET/train.csv')
print(df1)

df2=pd.read_csv('C:/DATASCIENCE/MyProjects/My_Projects/DATASET/test.csv')
print(df2)

df=pd.concat([df1,df2],axis=0,ignore_index = True)
print(df)

print(df.info())
print(df.describe())
print(df.dtypes)
print(df.shape)

print(df.nunique())
print(df.isnull().sum())

df['trip_duration'] = df['trip_duration'].astype(int)
df['num_of_passengers'] = df['num_of_passengers'].astype(int)
df['total_fare'] = df['total_fare'].astype(int)

data= df[df['fare']>0]
print(data)

#sns.countplot('num_of_passengers',data=data,color='cyan')
#plt.show()

sns.scatterplot(x='trip_duration', y='fare', data=data)
plt.show()

data1=data.loc[(data['trip_duration']>2000) & (data['num_of_passengers']==2)]
print(data1)

plt.figure(figsize=(10,6))
sns.lineplot(x='trip_duration',y='total_fare',data=data1,color='brown')
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
sns.distplot(data1['fare'],color='darkred')
plt.grid()
plt.show()

sns.pairplot(data1,height=2.5)
plt.show()

data1=data1.drop('num_of_passengers',axis=1)

plt.subplots(figsize=(10,5))
sns.heatmap(data1.corr(),annot=True, cmap="YlGnBu")
plt.show()

x= data.iloc[:,[0,1,2,3,4,5,7]].values
print(x)

y= data.iloc[:,6].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score

acc=r2_score(y_test,y_pred)
print(acc*100)

y_pred1=regressor.predict([[1348,4.75,3,85.00,28,12.00,0]])
print(y_pred1)