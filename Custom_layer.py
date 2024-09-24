import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Custom Dense Layer Implementation
class Dense(layers.Layer):
    def __init__(self, units, input_dim):
        super(Dense, self).__init__()
        self.w = self.add_weight(
            name='W',
            shape=(input_dim, units),
            initializer='random_normal',
            trainable=True,
        )
        
        self.b = self.add_weight(
            name='b',
            shape=(units,),
            initializer='zeros',  # Corrected initializer name
            trainable=True,
        )
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Reshape labels to 1D arrays
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Define the custom model
class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.flatten = layers.Flatten()  # Flatten layer to convert 3D input to 1D
        self.dense1 = Dense(64, 32*32*3)  # Updated input_dim for CIFAR-10
        self.dense2 = Dense(num_classes, 64)  # Number of units in dense1 layer
        
    def call(self, input_tensor):
        x = self.flatten(input_tensor)  # Flatten the input
        x = self.dense1(x)
        return self.dense2(x)

# Create and compile the model
model = MyModel()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)

# Evaluate the model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# Save the model's weights
model.save_weights('my_checkpoint.weights.h5')
