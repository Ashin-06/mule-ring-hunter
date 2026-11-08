import os
import sys

if sys.platform == "darwin":
    # PyTorch and XGBoost each bring their own OpenMP runtime on macOS, and two runtimes running
    # thread pools in one process crash or hang. One OpenMP thread avoids that, at some speed cost.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch  # noqa: F401
    import xgboost  # noqa: F401
