import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Define the model using the Functional API
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Convolutional layer with Batch Normalization and ReLU activation
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    
    # Additional convolutional layers
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Flatten the output and add dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)  # Output layer for 10 classes
    
    # Build the model
    model = keras.Model(inputs, outputs)
    return model

# Instantiate the model
model = my_model()

# Print the model architecture
model.summary()

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']  # Lowercase 'accuracy'
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

# Evaluate the model on the test data
