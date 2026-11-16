from pathlib import Path

RANDOM_SEED = 42

CLASSES = [
    "airplane", "apple", "bicycle",
    "car", "cat", "clock",
    "dog", "fish", "house", "tree"
]

DATA_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
BMC_PATH = BASE_DIR / "ckpts"

class Model:
    def __init__(self):
        pass