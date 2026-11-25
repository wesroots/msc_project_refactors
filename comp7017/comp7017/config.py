from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from keras.callbacks import ModelCheckpoint

import numpy as np

RANDOM_SEED = 42

CLASSES = [
    "airplane", "apple", "bicycle",
    "car", "cat", "clock",
    "dog", "fish", "house", "tree"
]

DATA_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

# ----- Directories -----
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
BMC_PATH = BASE_DIR / "ckpts"

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