#ANN

#part 1- Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
#creating independent feature
X=dataset.iloc[:,3:13]
#creating dependent feature
y=dataset.iloc[:,13]

#creating dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X["Gender"],drop_first=True)

# concatenate the data frame
X=pd.concat([X,geography,gender],axis=1)

#drop unnecessary columns(whose dummies variables have been created)
X=X.drop(["Geography","Gender"],axis=1)

#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train ,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Part 2- making ANN

#importing keras libraries and packages
import keras
from keras.models import Sequential # responsible for creating any NN library
from keras.layers import Dense 
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


#initialising the ANN
classifier=Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer="he_uniform",activation="relu",input_dim=11))

#adding hidden layer
classifier.add(Dense(units=6,kernel_initializer="he_uniform",activation="relu"))

#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer="glorot_uniform",activation="sigmoid"))

#compiling the ANN
classifier.compile(optimizer="Adamax",loss="binary_crossentropy",metrics=['accuracy'])

#fitting the ANN to the training set
model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)

#list all the data in history
print(model_history.history.keys())

#summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#Part 3- making the predictions and evaluating the model
#Predicting the Test set results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#calculate the accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

