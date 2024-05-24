import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data1=pd.read_csv('C:/DATASCIENCE/DATASET/CSV/Airline passenger satisfaction/train.csv')
print(data1)


data2=pd.read_csv('C:/DATASCIENCE/DATASET/CSV/Airline passenger satisfaction/test.csv')
print(data2)

data=pd.concat([data1,data2],axis=0,ignore_index = True)
print(data)

data=data.drop('Unnamed: 0',axis=1)

print(data.info())

print(data.shape)

print(data.describe())

print(data.dtypes)

print(data.nunique())

print(data.isnull().sum())

print(data.dropna(inplace=True))

data['Arrival Delay in Minutes']=data['Arrival Delay in Minutes'].astype(int)


plt.figure(figsize=(8,6))
sns.histplot(data=data,x="Age",binwidth=3,kde=True)
plt.show()

sns.countplot(x='Customer Type',data=data,color='brown')
plt.show()

sns.countplot(y='Type of Travel',data=data,color='crimson')
plt.show()


data['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Flight Classes')
plt.legend(title = "Classes:",loc='upper left')
plt.show()

sns.distplot(data['Flight Distance'],color='crimson')
plt.grid()
plt.show()


plt.figure(figsize=(8,6))
#sns.countplot('Class',hue='Gender',data=data,palette='flare')
#plt.show()


plt.figure(figsize=(6,6))
#sns.countplot('satisfaction',hue='Gender',data=data,palette='RdYlBu')
#plt.show()


a=data[data['Class']=='Eco Plus']


sns.lineplot(x='Online boarding',y='Inflight entertainment',data=a,color='red')
plt.show()

from sklearn.preprocessing import LabelEncoder

label_encoder =LabelEncoder()
data['satisfaction']= label_encoder.fit_transform(data['satisfaction'])

a['age_group'] = pd.cut(a.Age, bins=[0, 29, 40], right=True, labels=['under 40','over 40'])

plt.figure(figsize=(6,6))
sns.countplot(x='satisfaction', data=a, hue='age_group')
plt.show()

sns.violinplot(x='Inflight wifi service',y='Ease of Online booking',data=a,color='red')
plt.show()

plt.subplots(figsize=(15,10))
sns.heatmap(data.corr(),annot=True, cmap="YlGnBu")
plt.show()


data=data.drop(['Departure Delay in Minutes','Arrival Delay in Minutes'],axis=1)


categorical_columns = data.select_dtypes(['object']).columns

label_encoder =LabelEncoder()
for col in data[categorical_columns]:
    data[col]= label_encoder.fit_transform(data[col])

print(data)

x=data.iloc[:,:-1]
x

x.columns


y=data.iloc[:,-1]
y

#from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest,chi2

X_clf_new=SelectKBest(score_func=chi2,k=12).fit_transform(x,y)

print(X_clf_new[:5])

X=data.iloc[:,[0,2,3,4,5,6,7,9,11,12,13,14,17,21]]

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


#Model Development and Prediction
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)


# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
df2=pd.DataFrame(X_test)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df2


#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


y_pred1=logreg.predict([[92237,1,45,0,2,850,4,3,5,2,4,2,4,5]])
print(y_pred1)



