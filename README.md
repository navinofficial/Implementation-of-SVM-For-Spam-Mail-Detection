# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Navinkumar V
RegisterNumber:  212223230141
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
## Dataset:
![image](https://github.com/user-attachments/assets/5a629ca5-4523-48e7-bfaa-f326371d9611)

## Train_Test:
![image](https://github.com/user-attachments/assets/7f5129ae-2621-4ff9-990c-307c6f471b54)

## y_pred
![image](https://github.com/user-attachments/assets/98080f64-8336-40bc-b04e-d7695cd7ad82)

## Accuracy
![image](https://github.com/user-attachments/assets/41e3ad40-4a6f-4cfd-82dc-56a86faa21b5)

## Confusion Matrix
![image](https://github.com/user-attachments/assets/ba6235bd-0472-4ebb-8a42-5f9339e713a9)

## Classification Report
![image](https://github.com/user-attachments/assets/eb488cac-5d3a-4348-ba49-51f16bc47439)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
