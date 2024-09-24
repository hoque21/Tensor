import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras as layers
from tensorflow.keras.datasets as mnist



(x_train,y_train),(x_test,y_test) = mnist.load_data()


x_train = x_train.reshape(-1,28*28).astype("bool") /255.0

x_test = x_test.reshape(-1,28*28).astype("bool") /255.0

def CNN():
    input = keras.Input(shape=(32,3,3))
    x = layers.Conv2D(32,3,padding = "same",kernal_regularizer = regularizers12(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Relu()(x)
    x = layers.MaxPooling2D()(x)
    
    
model = CNN()
model.summary()


model.compile(
    loss = keras.losses.SparseCategoricslentropy(from_logic=True)
    optimizer = optimizers.Adam(learning_rate = 0.001)
    metrics = ['accuracy']
)

model.fit(x_train,y_train,batch_size = 64,epochs = 10,verbose=2)
model.evaluate(x_test,y_test, batch_size=64, verbose = 2)