import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Display some sample images from the training dataset


# Print dataset information

# Normalize the images
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

# Data augmentation function
def augment(image, label):
    # Resize to a larger size for augmentation
    image = tf.image.resize(image, (32, 32))

    # Apply random brightness and contrast adjustments
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.4)

    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)

    # Resize back to the original input size expected by the model (28x28)
    image = tf.image.resize(image, (28, 28))

    return image, label


AUTOTUNE = tf.data.AUTOTUNE  # Use tf.data.AUTOTUNE
BATCH_SIZE = 128

# Processing the training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()  # Cache the data for faster performance
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)  # Shuffle the dataset
ds_train = ds_train.batch(BATCH_SIZE)  # Batch the data
ds_train = ds_train.prefetch(AUTOTUNE)  # Prefetch for performance

# Processing the test dataset (without augmentation)
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(32)  # Batch size for test dataset
ds_test = ds_test.prefetch(AUTOTUNE)  # Prefetch for performance

# Define the model
model = keras.Sequential([
    keras.Input((28, 28, 1)),  # Input shape matches the final shape of augmented data
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='softmax'),
    layers.MaxPooling2D(pool_size=(2,3)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10),
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train the model
model.fit(ds_train, epochs=5, verbose=2)

# Evaluate the model on the test dataset
model.evaluate(ds_test)
