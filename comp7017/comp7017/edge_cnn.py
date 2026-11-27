import keras
from keras import layers

from comp7017.kernels import generate_prior_kernels, generate_random_kernels, construct_initializer

def _build_m4(X_train, initializer, normalizer):

    normalizer.adapt(X_train)

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(784,)),

        # Norm. layer
        normalizer,

        # Reshape layer
        layers.Reshape((28,28,1)),

        # Filter section 1
        layers.Conv2D(
            filters=32, kernel_size=3,
            kernel_initializer=initializer, trainable=False,
            padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2,2)),

        # Filter section 2
        layers.Conv2D(
            filters=32, kernel_size=3,
            kernel_initializer=initializer, trainable=False,
            padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
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