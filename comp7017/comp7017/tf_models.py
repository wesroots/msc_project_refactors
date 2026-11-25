import tensorflow as tf
import keras
from keras import layers

# Model 0 -> Baseline ANN
# One 64-neuron hidden layer followed by an output layer
def _build_m0(X_train):

    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(784,)),
        # Norm. layer
        normalizer,
        # Small dense layer
        layers.Dense(units=64, activation="relu"),
        # Output layer
        layers.Dense(units=10, activation="softmax")
    ], name="baseline_ann")

    return model

# Model 1 -> First CNN
# One convolutional layer
def _build_m1(X_train):
    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(784,)),
        # Norm. layer
        normalizer,
        # Reshape layer
        layers.Reshape((28,28,1)),
        # Filter + pooling layer
        layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),
        # Output layer
        layers.Flatten(),
        layers.Dense(units=10, activation="softmax")
    ], name="single_conv_cnn")

    return model

# CNN2 -> Iterative refinement
# Additional convolutional layer
def _build_m2(X_train):

    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(784,)),
        # Norm. layer
        normalizer,
        # Reshape layer
        layers.Reshape((28,28,1)),
        # Filter + pooling layer 1
        layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),
        # Filter + pooling layer 2
        layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),
        # Output layer
        layers.Flatten(),
        layers.Dense(units=10, activation="softmax")
    ], name="double_conv_cnn")

    return model

# CNN3 -> iterative refinement
# Addition of a small ANN after filter + pooling layers
def _build_m3(X_train):

    normalizer = layers.Normalization(axis=None)
    normalizer.adapt(X_train)

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(784,)),
        # Norm. layer
        normalizer,
        # Reshape layer
        layers.Reshape((28,28,1)),
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

    return model

# Compile model
def compile_model(model, loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=None):
    print("Compiling model", end=" ", flush=True)
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print("Done.")

# Build models
def build_models(X_train):
    for builder in (_build_m0, _build_m1, _build_m2, _build_m3):
        yield builder(X_train)