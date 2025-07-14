import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers,models,utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('btc_2015_2024.csv')

data = data.iloc[:, 1:]

data = data.to_numpy()

X = data[:,:-1]
y = data[:,-1]

print(X[-1][0])

X_train,_X,y_train,_y = train_test_split(X,y,test_size = 0.4,random_state = 1)

X_cv,X_test,y_cv,y_test = train_test_split(_X,_y,test_size = 0.5,random_state = 1)

del _X,_y



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_cv_scaled = scaler.transform(X_cv)
X_test_scaled = scaler.transform(X_test)

utils.set_random_seed(812)

model = models.Sequential([
    layers.Dense(25,activation = 'relu',name = 'nivel1'),
    layers.Dense(15,activation = 'relu',name = 'nivel2'),
    layers.Dense(10,activation = 'relu',name = 'nivel3'),
    layers.Dense(1,activation = 'linear',name = 'nivel4'), 
])

model.compile(
    optimizer = 'adam',
    loss = 'mse'
)

model.fit(
    X_train_scaled,
    y_train,
    epochs = 300
)

my_input = np.array([[117530,118219,116.977,117435,45.52e9,100,60.47,70.1,70.41,106692,107216,99257,102877,865,109.811,20599,2942,2942]])

my_input_scaled = scaler.transform(my_input)

aux = model.predict(my_input_scaled)

print(aux)