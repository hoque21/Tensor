import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# Load IMDB Reviews dataset
(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
)

# Tokenizer: Text vectorization layer to turn text into sequences of tokens
VOCAB_SIZE = 10000
encoder = layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(ds_train.map(lambda text, label: text))

# Normalize text: Convert text into sequences of tokens
def preprocess_text(text, label):
    text = encoder(text)
    return text, label

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

# Preprocess the training dataset
ds_train = ds_train.map(preprocess_text, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Preprocess the test dataset
ds_test = ds_test.map(preprocess_text, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

# Define a Text Classification model using Conv1D for sequence data
model = tf.keras.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64),  # Embedding layer
    layers.Conv1D(32, 3, activation='relu'),                # Conv1D layer for text
    layers.GlobalMaxPooling1D(),                            # Global max pooling
    layers.Dense(64, activation='relu'),                    # Fully connected layer
    layers.Dense(1)                                         # Output layer (binary classification)
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train the model
model.fit(ds_train, epochs=5, verbose=2)

# Evaluate the model
model.evaluate(ds_test)
