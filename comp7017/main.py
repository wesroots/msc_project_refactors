from comp7017.data import load_classes, prepare_xy_splits, prepare_ttv_splits, preprocess_X, preprocess_y

# Load data
print("<<< PREPARING DATA >>>")
load_classes()

X, y = prepare_xy_splits()

ttv_splits = prepare_ttv_splits(X, y)

X_train_pp, X_test_pp, X_val_pp = preprocess_X(
    ttv_splits["X_train"],
    ttv_splits["X_test"],
    ttv_splits["X_val"]
)

y_train_pp, y_test_pp, y_val_pp = preprocess_y(
    ttv_splits["y_train"],
    ttv_splits["y_test"],
    ttv_splits["y_val"]
)

# print("<>")