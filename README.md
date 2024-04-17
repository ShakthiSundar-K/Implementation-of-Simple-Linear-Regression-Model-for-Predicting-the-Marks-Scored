# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the
given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv('/content/ML_02_data.csv')
df.head()
df.tail()
#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying predicted values
y_pred
#displaying actual values
y_test
#graph plot for training data
plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

### df.head():
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/232d67d4-7d0f-46b9-8e6d-76a38cd91043)
### df.tail():
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/1e1e9820-2e87-42f7-8806-b60724790cf8)
### Array value of X:
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/8183d7f1-b892-402e-9ee6-74f8543d8ce2)
### Array value of Y:
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/163953da-f3cb-4395-bbb8-d7b71e3acbf9)
### Values of Y prediction:
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/669b908f-1bd8-410c-90f6-72650ecd82b5)
### Values of Y test:
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/b2f215bd-864e-43d5-a867-6391f6abc610)
### Training Set Graph:
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/ebc72121-461d-46a6-8718-5db58ec289f4)
### Test Set Graph:
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/82927cac-0360-4ea3-ac93-1d81f6d14f3f)
### Values of MSE,MAE and RMSE:
![image](https://github.com/DhanushPalani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121594640/00b28b13-7418-4b6a-9fb6-1edb157a9580)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
