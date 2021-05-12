import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score




iris = datasets.load_iris()
#split it in features and labels
X = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

#split them in train(20%) and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
""" 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

 """

model = svm.SVC()
model.fit(X_train, y_train)

#print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

#print("Predictions:", predictions)
#print("actual:", y_test)

""" 
for i in range(len(predictions)):
    print(classes[predictions[i]], " | ", classes[y_test[i]])

 """


#############################
#                           #
#           KNN             #
#                           #
#############################
""" 
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt


data = pd.read_csv('car.data')
#print(data.head())
#we're gonna use only 'buying', 'maint' and 'safety' as features
X = data[['buying', 'maint', 'safety']].values
y = data[['class']]

#let's convert those strings to numbers
X = np.array(X)
#print(X)

#converting data
#X
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

#print(X)

#for y we do a different type of conversion (mapping)

label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)

#let's create the model now!

knn = svm.SVC()

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print("predictions:", prediction)
print("accuracy: ", accuracy*100)

a = 1711

print("actual value ", y[a])
print("predicted value", knn.predict(X)[a])

 """

##################################
#                                #
#        Linear Regression       #
#                                #
##################################



from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



boston = datasets.load_boston()

#featres / labels
X = boston.data
y = boston.target

# print("X")
# print(X)
# print(X.shape)
# print("y")
# print(y)
# print(y.shape)



#algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[5], y)
plt.show()

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train and model creation
model = l_reg.fit(X_train, y_train)

predictions = model.predict(X_test)
print("predictions: ", predictions)
print("R^2: ", l_reg.score(X, y)) #???
print("coeff: ", l_reg.coef_) #slope
print("intercept: ", l_reg.intercept_) #???











