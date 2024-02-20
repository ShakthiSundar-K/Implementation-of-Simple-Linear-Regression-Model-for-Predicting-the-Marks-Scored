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
from sklearn.metrics import mean_absolute_error,mean_squared_error
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

###df.head()
![image](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/533f021b-e862-44d0-8583-50ceeb1236ca)
###df.tail()
![image](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/95b67a4e-cbc7-420e-9519-0994a2dc6c4a)
###Array value of X:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/bc82b991-107e-4227-8274-98e001f4becb)
###Array value of Y:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/bafe4d60-1b37-4185-9476-b2452523f59e)
###Values of Y prediction:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/b2df0322-eb2b-424f-aa91-ea735eaff19e)
###Values of Y test:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/6ec5e0ee-e316-4e34-8631-bd1ecf5f0e5a)
###Training Set Graph and Test Set Graph:
![Untitled](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/f4bb444e-c18b-4845-bcaa-68eb82d96f9a)
###Values of MSE, MAE and RMSE:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116143/12d3790a-21e2-43bf-869d-e52e7d3abc33)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
