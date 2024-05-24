import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


data=pd.read_csv('C:\DATASCIENCE\MyProjects\My_Projects\DATASET\heart_disease_dataset.csv')
print(data)

print(data.info())

print(data.describe())

print(data.shape)

print(data.dtypes)

print(data.nunique())

print(data.isnull().sum())

#sns.countplot('ChestPainType',hue='Sex',data=data)
#plt.show()

#sns.countplot('RestingECG',hue='HeartDisease',data=data)
#plt.show()

sns.barplot(x='Sex',y='HeartDisease',data=data,palette='YlGnBu_r')
plt.show()

from sklearn.preprocessing import LabelEncoder

categorical_columns = data.select_dtypes(['object']).columns

label_encoder =LabelEncoder()
for col in data[categorical_columns]:
    data[col]= label_encoder.fit_transform(data[col])

print(data)

for i  in data:
    sns.boxplot(data[i])
    plt.show()

plt.subplots(figsize=(20,10))
sns.heatmap(data.corr(),annot=True, cmap="YlGnBu")

print(data)

x=data.iloc[:,:-1].values
print(x)

y=data.iloc[:,11].values
print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier

classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print(df2)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_pred)

print(acc*100)

y_pred1=classifier.predict([[50,0,2,160,249,1,1,162,0,0.0,2]])

print(y_pred1)






