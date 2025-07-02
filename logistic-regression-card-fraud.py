import numpy as np
import pandas as pd

def sigmoid(z):
    return  1 / (1 + np.exp(-z))

def cost_function(X,y,w,b):
    cost = 0
    epsilon = 0.00001
    lambda_ = 0.01
    m = X.shape[0]
    z = np.dot(X,w) + b
    cost = -y * np.log(sigmoid(z) + epsilon) - (1 - y) * np.log(1 - sigmoid(z) + epsilon)
    cost = np.sum(cost) / m
    reg = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg
def compute_gradient(X,y,w,b):
    m,n = X.shape
    lambda_ = 0.01
    dj_dw = np.zeros(n)
    dj_db = 0

    z = np.dot(X,w) + b
    dj_dw =  np.dot(X.T, sigmoid(z) - y)  + (lambda_ / m) * w
    dj_db = np.sum(sigmoid(z) - y)

    dj_db /= m
    dj_dw /= m

    return dj_dw,dj_db
def gradient_descent(X,y,alpha = 0.01,iterations = 1000):
    m = X.shape[1]
    w = np.zeros(m)
    b = 0
    for i in range(iterations):
        dj_dw,dj_db = compute_gradient(X,y,w,b)
        w = w  - alpha * dj_dw
        b = b - alpha * dj_db
        print(cost_function(X,y,w,b))
    return w,b

def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def predict(X,w,b,y):
    for i in range(X.shape[0]):
        y[i] =  1 if sigmoid(np.dot(X[i],w) + b) >= 0.5 else 0
    return y

def feature_mapping(X, degree=2):
    m, n = X.shape
   
    out = [X] 
    for i in range(n):
        out.append(X[:, i:i+1] ** 2) 
    for i in range(n):
        for j in range(i, n):
            out.append(X[:, i:i+1] * X[:, j:j+1])
    return np.hstack(out)

arr = pd.read_csv("creditcard.csv",skiprows = 0)  

matrix = np.array(arr)

y_train = matrix[:,-1]
X_train = matrix[:,:len(matrix[0])-1]


X_mapped= feature_mapping(X_train)

X = standardize(X_mapped)



w ,b = gradient_descent(X,y_train,0.1)



yprime = np.zeros(len(y_train))

yprime = predict(X,w,b,yprime)



print( np.mean(yprime == y_train) * 100)