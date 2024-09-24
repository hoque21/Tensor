import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import pandas as pd

# CNN Block Definition
class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernal_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernal_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., tf.newaxis].astype("float32") / 255
x_test = x_test[..., tf.newaxis].astype("float32") / 255

# Model Definition
model = keras.Sequential(
    [
        CNNBlock(32),
        CNNBlock(64),
        CNNBlock(128),
        layers.Flatten(),
        layers.Dense(10)
    ]
)

# Model Compilation
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Model Training
model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=2)

# Model Evaluation
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
