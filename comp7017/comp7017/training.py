from keras.callbacks import ModelCheckpoint
from comp7017.config import BMC_PATH
from pathlib import Path

import time

BATCH_SIZE = 32
EPOCHS = 15
VERBOSE = True

def create_callback(name: str, monitor: str = "val_accuracy", path: Path = BMC_PATH) -> tuple[ModelCheckpoint, Path]:

    print(f"Creating callback for model '{name}'...", end=" ", flush=True)

    path.mkdir(parents=True, exist_ok=True)

    ckpt_file_path = path / f"checkpoint.{name}.keras"

    checkpoint = ModelCheckpoint(
        filepath=ckpt_file_path,
        monitor=monitor,
        mode="max",
        save_best_only=True
    )

    print("Done.")

    return (checkpoint, ckpt_file_path)

def fit_model(
        model, X_train, y_train, X_val, y_val, bmc_callback: ModelCheckpoint,
        batch_size=32, epochs=15, verbose=True
    ):

    start_time = time.perf_counter()

    hist = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=(X_val, y_val),
        callbacks=bmc_callback
    )

    train_time = time.perf_counter() - start_time
    print(f"Elapsed time: {train_time // 60}m {train_time % 60:.0f}s")

    return hist, train_time