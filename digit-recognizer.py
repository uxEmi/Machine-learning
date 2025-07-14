import numpy as np
import pandas as pd
from keras import models,layers,utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras

def f(r):
    y = np.zeros(len(r))
    for i in range(len(r)):
        y[i] = np.argmax(r[i])
    return y
dataframe = pd.read_csv("train.csv",skiprows = 0)
dataframetest = pd.read_csv("test.csv",skiprows = 0)


X = dataframe.to_numpy()
Xt = dataframetest.to_numpy()

y = X[:,0]
X = X[:,1:]

X_train,X_cv,y_train,y_cv = train_test_split(X,y,test_size=0.4,random_state=1)

model = models.Sequential([
    layers.Dense(100,activation = 'relu',name = 'layer1'),
    layers.Dense(50,activation = 'relu',name = 'layer2'),
    layers.Dense(10,activation = 'softmax',name = 'layer3')
]
)

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy()    
)

model.fit(
    X_train,
    y_train,
    epochs= 10
)

yhat = model.predict(X_cv)

yhat = f(yhat)

print(np.mean(yhat == y_cv)*100)