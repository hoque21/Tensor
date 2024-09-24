import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

# TensorFlow Hub থেকে প্রি-ট্রেইনড মডেল লোড করা (ResNetV2 এর ফিচার এক্সট্রাক্টর)
hub_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

# Sequential মডেলে KerasLayer এবং Keras এর অন্যান্য লেয়ার যুক্ত করা
model = keras.Sequential([
    hub.KerasLayer(hub_url, input_shape=(224, 224, 3), trainable=False),  # ট্রেনিং বন্ধ রাখা
    layers.Dense(128, activation='relu'),  # Keras এর Dense লেয়ার যুক্ত করা
    layers.Dropout(0.5),  # ওভারফিটিং কমানোর জন্য Dropout লেয়ার
    layers.Dense(10, activation='softmax')  # ফাইনাল ক্লাসিফায়ার লেয়ার (১০টি ক্লাসের জন্য)
])

# মডেল কম্পাইল করা
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# মডেল সারাংশ প্রিন্ট করা
print(model.summary())
