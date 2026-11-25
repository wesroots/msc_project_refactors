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

from comp7017.config import RANDOM_SEED, CLASSES, DATA_BASE_URL, DATA_DIR, BMC_PATH

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_classes():
    print("Downloading classes...", end=" ", flush=True)
    n_new_dl = 0
    n_existing_dl = 0
    for cls in CLASSES:
        filepath = DATA_DIR / f"full_numpy_bitmap_{cls}.npy"
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(DATA_BASE_URL + cls + ".npy", filepath)
            n_new_dl += 1
        else:
            n_existing_dl += 1
    print("Done.")
    print(f"{n_new_dl} file(s) downloaded; {n_existing_dl} already downloaded.\n")

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