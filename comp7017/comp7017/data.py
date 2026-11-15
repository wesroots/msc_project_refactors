import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import tensorflow as tf
import urllib

from pathlib import Path
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from comp7017.config import RANDOM_SEED, CLASSES, DATA_BASE_URL, DATA_DIR, BMC_PATH

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_classes():
    print("Downloading classes...")
    for cls in CLASSES:
        filepath = DATA_DIR / f"full_numpy_bitmap_{cls}.npy"

        if not os.path.exists(filepath):
            print(f" - Downloading {cls}...", end=" ", flush=True)
            urllib.request.urlretrieve(DATA_BASE_URL + cls + ".npy", filepath)
            print("Done.")
        else:
            print(f" - 'full_numpy_bitmap_{cls}.py' already downloaded.")
    print("\nAll files downloaded successfully.\n")

def prepare_xy_splits() -> tuple[NDArray, NDArray]:
    print("Preparing X/y splits...", end=" ", flush=True)
    X_list = []
    y_list = []

    for label, cls in enumerate(CLASSES):
        data = np.load(DATA_DIR / f"full_numpy_bitmap_{cls}.npy")
        data = data[:7500]

        X_list.append(data)

        labels = np.full(len(data), label)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print("Done.")
    print(f" - X: {X.shape}, y: {y.shape}\n")

    return (X, y)

def prepare_ttv_splits(X: NDArray, y: NDArray):
    print("Preparing train/test/val splits...", end=" ", flush=True)

    # Split dataset into stratified train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Split 20% of training set for validation set
    X_train, X_val, y_train, y_val =  train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    ttv_splits = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val
    }

    print("Done.")

    for label, split in ttv_splits.items():
        print(f" - {label}: {split.shape}")

    print("")

    return ttv_splits

def preprocess_X(X_train, X_test, X_val):

    print("Preprocessing inputs...", end=" ", flush=True)

    X_train_pp = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_test_pp = np.reshape(X_test, (len(X_test), 28, 28, 1))
    X_val_pp = np.reshape(X_val, (len(X_val), 28, 28, 1))

    print("Done.")

    print(f" - X_train_pp: {X_train_pp.shape}")
    print(f" - X_test_pp: {X_test_pp.shape}")
    print(f" - X_val_pp: {X_val_pp.shape}\n")

    return X_train_pp, X_test_pp, X_val_pp

def preprocess_y(y_train, y_test, y_val):
    print("Preprocessing outputs...", end=" ", flush=True)

    y_train_rs = y_train.reshape(-1, 1)
    y_test_rs = y_test.reshape(-1, 1)
    y_val_rs = y_val.reshape(-1, 1)

    # Fit OHE and transform train/test/val output sets
    ohe = OneHotEncoder(sparse_output=False)

    ohe.fit(y_train_rs)

    y_train_pp = ohe.transform(y_train_rs)
    y_test_pp = ohe.transform(y_test_rs)
    y_val_pp = ohe.transform(y_val_rs)

    print("Done.")
    print(f" - y_train_pp: {y_train_pp.shape}")
    print(f" - y_test_pp: {y_test_pp.shape}")
    print(f" - y_val_pp: {y_val_pp.shape}\n")

    return y_train_pp, y_test_pp, y_val_pp