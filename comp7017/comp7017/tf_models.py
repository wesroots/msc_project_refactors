import tensorflow as tf
import keras
from keras import layers

from config import Model

model_0 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Normalization(),
    layers.Dense(units=64, activation="relu"),
    layers.Dense(units=10, activation="softmax")
])

# CNN1
model_1 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Normalization(),
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dense(units=10, activation="softmax")
])