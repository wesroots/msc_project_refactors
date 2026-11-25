# COMP7017 &mdash; Deep Learning: Mini Project
This project was originally developed as part of my MSc Data Science coursework. It is currently being refactored to improve maintainability, reproducibility, computational correctness and extensibility.

---

### Improvements:
| Improvement  | Original implementation | Improved implementation |
| ------------- | ------------- | ------------- |
| Input preprocessing  | Separate reshaped datasets for ANN and CNN architectures | Reshaping incorporated into the model architecture, enabling a consistent input format |
| Normalisation | Normalisation layer without adaptation | Explicit training-data-adapted normalisation |
| Target encoding | One-hot encoded class labels | Integer labels with sparse categorical cross-entropy |
| Code architecture | Notebook-based implementation | Modular Python package with separation of concerns |
| Model checkpointing | Individually defined callbacks | Reusable checkpoint function |
| Model training | Repeated training code | Reusable training function |
| Model evaluation | Repeated evaluation logic | Model container class for performance metrics and training metadata |