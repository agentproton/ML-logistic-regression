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

def Stochastic_Gradient(X,Y):
  m=len(Y)
  theta = np.zeros(3)
  learning_rate=0.01
  iterations=1500
  for i in range(iterations):
    for j in range(m):
      index=random.randint(0,m-1)
      x_value=X[index]
      y_value=Y[index]
      theta=theta-(learning_rate/m)*np.dot(x_value.T,(sigmoid(np.dot(x_value,theta)) - y_value))
        # print(theta
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

df = pd.read_csv('/content/sample_data/dataset_ass4.csv')
df.head()
a=df['marks1'].values
b=df['marks2'].values
c=df['value'].values
x=[]
y=[]
x1=[]
s=70
for i in range(s):
  y.append(c[i])
  x.append([1,a[i],b[i]])
  x1.append([1,a[i],b[i]])
x=np.array(x)
y=np.array(y)
x1=np.array(x1)
x1=scale(x1)

theta2=Stochastic_Gradient(x,y)
theta3=Stochastic_Gradient(x1,y)

print('THE PARAMETER VALUES ARE(without feature scaling in Stochastic_Gradient algorithm):')
print(theta2)
print('THE PARAMETER VALUES ARE(with feature scaling in Stochastic_Gradient algorithm):')
print(theta3)

error2=0
error3=0
x_test=[]
y_test=[]
x1=[]
for i in range(70,len(a)):
  x_test.append([1,a[i],b[i]])
  y_test.append(c[i])
  x1.append([1,a[i],b[i]])
x1=scale(x1)

y_pred2=predict(x,theta2)
y_pred3=predict(x1,theta3)

for i in range(len(y_pred)):
  if y_pred2[i]!=y_test[i]:
    error2=error2+1
  if y_pred3[i]!=y_test[i]:
    error3=error3+1

error2/=len(y_test)
error3/=len(y_test)

print('The Error in the Stochastic_Gradient algorithm without feature scaling is:')
print(error2*100)
print('The Error in the Stochastic_Gradient algorithm with feature scaling is:')
print(error3*100)
