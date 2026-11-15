from pathlib import Path

RANDOM_SEED = 42

CLASSES = [
    "airplane", "apple", "bicycle",
    "car", "cat", "clock",
    "dog", "fish", "house", "tree"
]

DATA_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

DATA_DIR = Path("../data")

BMC_PATH = Path("../data/ckpts")

class Model:
    def __init__(self):
        pass