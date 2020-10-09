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
  theta = np.zeros(9)
  learning_rate=0.01
  iterations=1500
  for i in range(iterations):
    theta=theta-(learning_rate/m) * np.dot(x.T,(sigmoid(np.dot(x,theta)) - y))
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
  x.append([1,a[i],b[i],a[i]**2,b[i]**2,a[i]*b[i],(a[i]**2)*b[i],a[i]*(b[i]**2),a[i]**3])
x=np.array(x)
y=np.array(y)
x=scale(x)
theta=gradient_descent(x,y)

print("THE PARAMETER VALUES ARE(with adding more features  in batch gradient descent algorithm):")
print(theta)

error=0

x_test=[]
y_test=[]

for i in range(70,len(a)):
  x_test.append([1,a[i],b[i],a[i]*a[i],b[i]*b[i],a[i]*b[i],(a[i]**2)*b[i],a[i]*(b[i]**2),a[i]**3])
  y_test.append(c[i])
x_test=scale(x_test)
y_pred=predict(x_test,theta)

for i in range(len(y_pred)):
  if y_pred[i]!=y_test[i]:
    error=error+1
 
error/=len(y_test)

print('The Error in the Gradient Descent Algorithm with adding more features is:')
print(error*100)

