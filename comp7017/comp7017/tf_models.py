import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
import numpy as np
from keras import layers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from keras.callbacks import ModelCheckpoint

# Model container class
class ModelContainer:
    def __init__(self, tf_model, history, train_time, X_test, y_test):
        self.tf_model = tf_model
        self.name = tf_model.name
        self.history = history.history
        self.train_time = train_time

        probs = self.tf_model.predict(X_test, batch_size=32)
        self.y_pred = np.argmax(probs, axis=1)

        self.test_loss, self.test_acc = self.tf_model.evaluate(X_test, y_test, verbose=0)

        self.precision = precision_score(y_test, self.y_pred, average="macro")
        self.recall = recall_score(y_test, self.y_pred, average="macro")
        self.f1 = f1_score(y_test, self.y_pred, average="macro")

        self.best_epoch = int(np.argmax(self.history["val_accuracy"])) + 1
        self.ckpt_val_acc = self.history["val_accuracy"][self.best_epoch - 1]
        self.ckpt_val_loss = self.history["val_loss"][self.best_epoch - 1]

    def __repr__(self):
        return f"{self.tf_model.name}"

    __str__ = __repr__

# Normalizer
normalizer = layers.Normalization(axis=None)

# Model 0 -> Baseline ANN
# One 64-neuron hidden layer followed by an output layer
def _build_m0(X_train, normalizer=normalizer):

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

# Model 1 -> first CNN
# One convolutional layer
def _build_m1(X_train, normalizer=normalizer):
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

# CNN2 -> iterative refinement
# Additional convolutional layer
def _build_m2(X_train, normalizer=normalizer):

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
def _build_m3(X_train, normalizer=normalizer):

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
    print("Compiling model...", end=" ", flush=True)
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print("Done.")

ALL_MODELS = False
from comp7017.edge_cnn import _build_m4
CURR_MODEL = _build_m4

# Build models
def build_models(X_train):
    if ALL_MODELS:
        for builder in (_build_m0, _build_m1, _build_m2, _build_m3):
            yield builder(X_train)
    else:
        return CURR_MODEL(X_train)