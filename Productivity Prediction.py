import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('C:\DATASCIENCE\MyProjects\My_Projects\DATASET\garments_worker_productivity.csv')
print(data)

data.info()
print(data.describe())
print(data.dtypes)
print(data.shape)
print(data.isnull().sum())

print(data.nunique())

data["date"]=pd.to_datetime(data["date"])
print(data.info())

print(data["day"].unique())

days = {"Sunday" : 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
data["day"] = [days[x] for x in data["day"]]
print(data['day'].value_counts())

print(data['wip'].value_counts())


data.dropna(inplace=True)
print(data.isnull().sum())

plt.figure(figsize=(10,5))
sns.lineplot(x='incentive',y='over_time',data=data,color='red')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='quarter',data=data,color='grey')
plt.show()

sns.distplot(data['incentive'],color='crimson')
plt.grid()
plt.show()

plt.figure(figsize=(9,6))
sns.histplot(data['targeted_productivity'],palette='Spectral',kde=True,binwidth=2)
plt.grid(axis='y',color='green',linestyle='--',linewidth=0.5)
plt.show()

plt.figure(figsize=(8,6))
explode=[0,0.1,0,0.1,0]
plt.pie(data['quarter'].value_counts(),labels=['quarter1','quarter2','quarter3','quarter4','quarter5'],
        explode=explode,autopct='%.0f%%')
plt.show()


plt.figure(figsize = (10, 10))
sns.heatmap(data.corr(), annot=True)
plt.show()

categorical_columns = data.select_dtypes(['object']).columns
from sklearn.preprocessing import LabelEncoder
label_encoder =LabelEncoder()
for col in data[categorical_columns]:
    data[col]= label_encoder.fit_transform(data[col])

data=data.drop('date',axis=1)

x=data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]]
print(x)
y=data.iloc[:,13]
print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result;
y_pred= regressor.predict(x_test)

df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
# predicting the accuracy score
score=r2_score(y_test,y_pred)
print("r2 score is ",score*100,"%")

















