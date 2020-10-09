import pandas as pd
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
import random

def minibatch_Gradient(X,Y):
  m=len(Y)
  theta = np.zeros(3)
  batch_size=20
  n_batches=int(m/batch_size)
  learning_rate=0.0001
  iterations=4000
  N = []
  r_factor=10
  for i in range(0, 3):
    N.append([0 if j != i or i == 0 else 1 for j in range(3) ])
  for i in range(iterations):
    idx=np.random.permutation(m)
    X=X[idx]
    Y=Y[idx]
    for j in range(0,m,batch_size):
      x_value=X[j:j+batch_size]
      y_value=Y[j:j+batch_size]
      theta=theta-(learning_rate/m)*np.dot(x_value.T,(sigmoid(np.dot(x_value,theta)) - y_value))-(((learning_rate*r_factor)/m)*np.dot(theta,N))
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
x1=[]
s=70
for i in range(s):
  y.append(c[i])
  x.append([1,a[i],b[i]])
  x1.append([1,a[i],b[i]])
x=np.array(x)
y=np.array(y)
x=scale(x)

theta4=minibatch_Gradient(x,y)

print('THE PARAMETER VALUES ARE(with regularization in minibatch_Gradient algorithm):')
print(theta4)

error4=0

x_test=[]
y_test=[]

for i in range(70,len(a)):
  x_test.append([1,a[i],b[i]])
  y_test.append(c[i])

x_test=scale(x_test)

y_pred4=predict(x_test,theta4)

for i in range(len(y_pred)):
  if y_pred4[i]!=y_test[i]:
    error4=error4+1

error4/=len(y_test)

print('The Error in the minibatch_Gradient algorithm with regularization is:')
print(error4*100)
