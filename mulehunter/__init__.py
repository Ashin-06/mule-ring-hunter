import sys

if sys.platform == "darwin":
    # PyTorch and XGBoost both use OpenMP on macOS. Loading PyTorch first lets XGBoost reuse the
    # runtime already in memory instead of loading a second copy, which crashes or hangs training.
    import torch  # noqa: F401
    import xgboost  # noqa: F401
