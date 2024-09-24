import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Display examples of the dataset (optional)
fig = tfds.show_examples(ds_train, ds_info, rows=3, cols=4)

# Normalize the images
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

# Preprocess training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Preprocess test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)

# Define CNN model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Input shape for MNIST (28x28 grayscale images)
    
    # First convolutional block
    layers.Conv2D(32, 3, activation='relu'),  # 32 filters, 3x3 kernel
    layers.MaxPooling2D(pool_size=(2, 2)),  # Downsample by 2
    
    # Second convolutional block
    layers.Conv2D(64, 3, activation='relu'),  # 64 filters, 3x3 kernel
    layers.MaxPooling2D(pool_size=(2, 2)),  # Downsample by 2
    
    # Flatten the feature maps into a 1D vector
    layers.Flatten(),
    
    # Fully connected (Dense) layer
    layers.Dense(128, activation='relu'),  # Dense layer with 128 units
    
    # Output layer with 10 units (for 10 classes) and no activation (because we'll use softmax later)
    layers.Dense(10),
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Use from_logits=True for unscaled outputs
    metrics=["accuracy"]
)

# Train the model
model.fit(ds_train, epochs=5, verbose=2)

# Evaluate the model on the test set
model.evaluate(ds_test)
