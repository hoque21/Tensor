import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32")  /255.0

model = keras.Sequential()
model.add(keras.Input(shape=(None,28)))
model.add(
    layers.Bidirectional(layers.LSTM(512, return_sequences=True , activation='relu'))
)
model.add(
    layers.Bidirectional(layers.LSTM(512, activation='relu'))
)
model.add(layers.Dense(10))

print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train,y_train, batch_size = 64, epochs = 5 , verbose=2)
model.fit(x_test,y_test,batch_size=64, verbose=2)







print('hello ! i am done')