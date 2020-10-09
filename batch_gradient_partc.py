import pandas as pd
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X,theta):
    return np.round(sigmoid(np.dot(X,theta)))

def gradient_descent(x,y):
  m=len(y)
  theta = np.zeros(3)
  learning_rate=0.01
  iterations=1500
  r_factor=10
  N=[]
  for i in range(0, 3):
    N.append([0 if j != i or i == 0 else 1 for j in range(3) ])
  for i in range(iterations):
    theta=theta-(learning_rate/m) * np.dot(x.T,(sigmoid(np.dot(x,theta)) - y))-(((learning_rate*r_factor)/m)*np.dot(theta,N))
  return theta
  
def scale(X):
  for col in range(len(X[0])):
    max_x = max([X[row][col] for row in range(len(X))])
    min_x = min([X[row][col] for row in range(len(X))])
    if min_x == max_x:
      continue
    for row in range(len(X)):
      X[row][col]= (X[row][col] - min_x)/ (max_x- min_x)
  return X

#reading data
df = pd.read_csv('/content/sample_data/dataset_ass4.csv')
df.head()
a=df['marks1'].values
b=df['marks2'].values
c=df['value'].values
x=[]
y=[]
s=70
for i in range(s):
  y.append(c[i])
  x.append([1,a[i],b[i]])
x=np.array(x)
y=np.array(y)
x=scale(x)
theta=gradient_descent(x,y)

print("THE PARAMETER VALUES ARE(with regularization in batch gradient descent algorithm):")
print(theta)

error=0

x_test=[]
y_test=[]

for i in range(70,len(a)):
  x_test.append([1,a[i],b[i]])
  y_test.append(c[i])
x_test=scale(x_test)
y_pred=predict(x_test,theta)

for i in range(len(y_pred)):
  if y_pred[i]!=y_test[i]:
    error=error+1
 
error/=len(y_test)

print('The Error in the Gradient Descent Algorithm with regularization is:')
print(error*100)

