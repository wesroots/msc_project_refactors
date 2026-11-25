import tensorflow as tf
import keras
from keras import layers

# Model 0 -> Baseline ANN
# One 64-neuron hidden layer followed by an output layer
m0 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Normalization(),
    layers.Dense(units=64, activation="relu"),
    layers.Dense(units=10, activation="softmax")
], name="Baseline ANN")
m0.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Model 1 -> First CNN
# One convolutional layer
m1 = keras.Sequential([
    # Input layer
    layers.Input(shape=(784,)),
    # Reshape layer
    layers.Reshape((28,28,1)),
    # Norm. layer
    layers.Normalization(),
    # Filter + pooling layer
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D((2,2)),
    # Output layer
    layers.Dense(units=10, activation="softmax")
], name="single_conv_cnn")
m1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# CNN2 -> Iterative refinement
# Additional convolutional layer
m2 = keras.Sequential([
    # Input layer
    layers.Input(shape=(784,)),
    # Reshape layer
    layers.Reshape((28,28,1)),
    # Norm. layer
    layers.Normalization(),
    # Filter + pooling layer 1
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D((2,2)),
    # Filter + pooling layer 2
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D((2,2)),
    # Output layer
    layers.Dense(units=10, activation="softmax")
], name="double_conv_cnn")
m2.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# CNN3 -> iterative refinement
# Addition of a small ANN after filter + pooling layers
m3 = keras.Sequential([
    # Input layer
    layers.Input(shape=(784,)),
    # Reshape layer
    layers.Reshape((28,28,1)),
    # Norm. layer
    layers.Normalization(),
    # Filter + pooling layer 1
    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2)),
    # Filter + pooling layer 2
    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2)),
    # Small regular ANN
    layers.Flatten(),
    layers.Dense(units=128, activation="relu"),
    layers.Dropout(0.45),
    layers.Dense(units=64, activation="relu"),
    # Output layer
    layers.Dense(units=10, activation="softmax"),
], name="cnn_dense_head")
m3.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])