import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

#dataset preprocessing done


(ds_train,ds_test), ds_info = tfds.load(
    "mnist",
    split=["train","test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

fig = tfds.show_examples(ds_train,ds_info,rows=3,cols = 4)

print(ds_info)

def normalize_img(image,label):
    return tf.cast(image, tf.float32)/255.0, label


AUTOTUNE = tf.data.AUTOTUNE  # Use tf.data.AUTOTUNE

BATCH_SIZE = 64


# Processing the training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()  # Cache the data for faster performance
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)  # Shuffle the dataset
ds_train = ds_train.batch(BATCH_SIZE)  # Batch the data
ds_train = ds_train.prefetch(AUTOTUNE)  # Prefetch for performance

# Processing the test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)  # Batch size for test dataset
ds_test = ds_test.prefetch(AUTOTUNE)  # Prefetch for performance



model = keras.Sequential([
    keras.Input((28,28,1)),
    layers.Conv2D(32,3,activation='relu'),
    layers.Flatten(),
    layers.Dense(10),
])

import tensorflow as tf

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]  # Use lowercase "accuracy"
)


model.fit(ds_train, epochs = 5,verbose=2)

model.evaluate(ds_test)