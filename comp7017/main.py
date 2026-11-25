from comp7017.data import load_classes, prepare_xy_splits, prepare_ttv_splits
from comp7017.tf_models import build_m3, compile_model
from comp7017.config import ModelContainer
from comp7017.training import create_callback, fit_model

from keras.models import load_model

# ---------- LOAD DATA ----------
print("<<< LOADING & PROCESSING DATA >>>")
load_classes()

X, y = prepare_xy_splits()

ttv_splits = prepare_ttv_splits(X, y)

# ---------- FIT MODEL ----------
model = build_m3(ttv_splits["X_train"])
compile_model(model)

print(f"<<< CONSTRUCTING & FITTING MODEL >>>")
callback, bmc_path = create_callback(model.name)
hist, train_time = fit_model(
    model=model,
    X_train=ttv_splits["X_train"],
    y_train=ttv_splits["y_train"],
    X_val=ttv_splits["X_val"],
    y_val=ttv_splits["y_val"],
    bmc_callback=callback
)

print("Loading best checkpoint weights...", end=" ", flush=True)
bmc_model = load_model(bmc_path)
print("Done")

model_container = ModelContainer(
    tf_model=bmc_model,
    history=hist,
    train_time=train_time,
    X_test=ttv_splits["X_test"],
    y_test=ttv_splits["y_test"]
)

