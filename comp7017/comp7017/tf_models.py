import tensorflow as tf
import keras
from keras import layers

# Model 0 -> Baseline ANN
# One 64-neuron hidden layer followed by an output layer
def build_m0(X_train):

    print("Building model...", end=" ", flush=True)

    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Normalization(),
        layers.Dense(units=64, activation="relu"),
        layers.Dense(units=10, activation="softmax")
    ], name="Baseline ANN")

    print("Done.")

    return model

# Model 1 -> First CNN
# One convolutional layer
def build_m1(X_train):

    print("Building model...", end=" ", flush=True)

    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
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

    print("Done.")

    return model

# CNN2 -> Iterative refinement
# Additional convolutional layer
def build_m2(X_train):

    print("Building model...", end=" ", flush=True)

    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
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

    print("Done.")

    return model

# CNN3 -> iterative refinement
# Addition of a small ANN after filter + pooling layers
def build_m3(X_train):

    print("Building model...", end=" ", flush=True)

    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(784,)),
        # Reshape layer
        layers.Reshape((28,28,1)),
        # Norm. layer
        normalizer,
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

    print("Done.")

    return model

# Compile model
def compile_model(model, loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]):
    print("Compiling model", end=" ", flush=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print("Done.")