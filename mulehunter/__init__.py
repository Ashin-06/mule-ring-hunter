import sys

if sys.platform == "darwin":
    # XGBoost and PyTorch each bring an OpenMP runtime on macOS. Loading XGBoost's first keeps
    # the two from clashing, which otherwise crashes or hangs training.
    import xgboost  # noqa: F401
