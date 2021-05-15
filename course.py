#############################
#                           #
#           Iris            #
#                           #
#############################

""" 
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
 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



model = svm.SVC()
model.fit(X_train, y_train)

#print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

#print("Predictions:", predictions)
#print("actual:", y_test)

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

""" 

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



 """

#############################
#                           #
#           NLP             #
#                           #
#############################

import json
import random

class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: #Score of 4 or 5
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
        
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

#Load data

file_name = './data/books_small_10000.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))
        
#print(reviews[5].text)

#Prepare data


from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)

test_container = ReviewContainer(test)


train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

#print(train_y.count(Sentiment.POSITIVE))
#print(train_y.count(Sentiment.NEGATIVE))



#Bag of words vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# This book is great !
# This book was so bad

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

#print(train_x[0])
#print(train_x_vectors[0].toarray())


###########################
#     Classification      #
###########################
from sklearn.metrics import accuracy_score, confusion_matrix


#Linear SVM
from sklearn import svm

clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)

test_x[0]

clf_svm.predict(test_x_vectors[0])

predictions = clf_svm.predict(test_x_vectors)
acc = accuracy_score(test_y, predictions)

print("Accurancy o SVM: ", round(acc*100,2), "%")

conf_matrix = confusion_matrix(test_y,predictions, labels=['POSITIVE','NEGATIVE'])

print("SVM Confusion Matrix")
print(conf_matrix)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

clf_dec.predict(test_x_vectors[0])


predictions = clf_dec.predict(test_x_vectors)
acc = accuracy_score(test_y, predictions)

print("Accurancy o Decision Tree: ", round(acc*100,2), "%")

conf_matrix = confusion_matrix(test_y,predictions, labels=['POSITIVE','NEGATIVE'])

print("SVM Confusion Matrix")
print(conf_matrix)

#Naive_bayes
from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.toarray(), train_y)

clf_gnb.predict(test_x_vectors[0].toarray())

predictions = clf_gnb.predict(test_x_vectors.toarray())
acc = accuracy_score(test_y, predictions)

print("Accurancy o Gaussian Naive Bayes: ", round(acc*100,2), "%")

conf_matrix = confusion_matrix(test_y,predictions, labels=['POSITIVE','NEGATIVE'])

print("SVM Confusion Matrix")
print(conf_matrix)

#Logistic Regression

from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

clf_log.predict(test_x_vectors[0])

predictions = clf_log.predict(test_x_vectors)
acc = accuracy_score(test_y, predictions)

print("Accurancy o Logisti Regression: ", round(acc*100,2), "%")

conf_matrix = confusion_matrix(test_y,predictions, labels=['POSITIVE','NEGATIVE'])

print("SVM Confusion Matrix")
print(conf_matrix)
#Passive Aggressive

from sklearn.linear_model import PassiveAggressiveClassifier

clf_pac=PassiveAggressiveClassifier(max_iter=50)
clf_pac.fit(train_x_vectors,train_y)

clf_pac.predict(test_x_vectors[0])

predictions = clf_log.predict(test_x_vectors)
acc = accuracy_score(test_y, predictions)

print("Accurancy of Passive Aggressive Classifier: ", round(acc*100,2), "%")

# F1 Scores
from sklearn.metrics import f1_score
print("F1 Scores:")
print("SVM: ", f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print("DEC: ", f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print("GNB: ", f1_score(test_y, clf_gnb.predict(test_x_vectors.toarray()), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print("LOG: ", f1_score(test_y, clf_log.predict(test_x_vectors.toarray()), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print("PAC: ", f1_score(test_y, clf_pac.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))


#TEST YOUT SELF
test_set = ['very fun', "bad book do not buy", 'I do not recommend it']
new_test = vectorizer.transform(test_set)

#print("SVM: ", clf_svm.predict(new_test))
#print("DEC: ", clf_dec.predict(new_test))
#print("GNB: ", clf_gnb.predict(new_test.toarray()))
#print("LOG: ", clf_log.predict(new_test.toarray()))



#####################################################
#                                                   #
#       Tuning our model (with Grid Search)         #
#                                                   #
#####################################################

""" 

from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)

predictions = clf.predict(test_x_vectors)
acc = accuracy_score(test_y, predictions)

print("Accurancy of gridded SVM: ", round(acc*100,2), "%")



#SAVE MODEL
import pickle

with open('./models/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

 """