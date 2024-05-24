import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('C:\DATASCIENCE\MyProjects\My_Projects\DATASET\Employee.csv')
print(data)

print(data.info())

print(data.shape)

print(data.describe())

print(data.dtypes)

print(data.nunique())

print(data.isnull().sum())

sns.lineplot(x='JoiningYear',y='PaymentTier',data=data,color='pink')
plt.show()

#sns.countplot('Education',data=data,color='maroon')
#plt.show()

sns.violinplot(x='Education',y='PaymentTier',data=data,palette='mako')
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(data=data,x="PaymentTier",kde=True,hue='Gender')
plt.show()

plt.subplots(figsize=(8,6))
sns.heatmap(data.corr(),annot=True, cmap="YlGnBu")
plt.show()

from sklearn.preprocessing import LabelEncoder

categorical_columns = data.select_dtypes(['object']).columns

label_encoder =LabelEncoder()
for col in data[categorical_columns]:
    data[col]= label_encoder.fit_transform(data[col])

print(data)

x = data.iloc[:,:-1].values  # independent variable
x

y = data.iloc[:,-1].values  # dependent
y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)



#Fitting K-NN classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
#n_neighbors: To define the required neighbors of the algorithm. Usually, it takes 5.
#metric='minkowski': This is the default parameter and it decides the distance between the points.
#p=2: It is equivalent to the standard Euclidean metric.
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )

classifier.fit(x_train, y_train)


#Predicting the test set result
y_pred= classifier.predict(x_test)
print(y_pred)

print("Prediction comparison")
df=pd.DataFrame({"Y_test":y_test,"Y-pred":y_pred})
df

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# evaluate predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


y_pred1= classifier.predict([[2,2018,2,3,47,0,1,8],[1,2013,3,1,32,1,1,5]])
print(y_pred1)






