"""Naive baselines: persistence + mean, scored on HELD-OUT subjects."""
from pathlib import Path
import numpy as np
from physiofusion.windowing import build_dataset
from physiofusion.splits import subject_grouped_split

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
SUBJECTS = [f"{i:03d}" for i in range(1, 17) if i != 15]

# build data + one split
X, y, groups = build_dataset(BIG, SUBJECTS)
train_idx, test_idx = subject_grouped_split(groups, n_splits=5, fold=0)

# split into train / test
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# --- error helpers (by hand, so you see the math) ---
def rmse(true, pred): return np.sqrt(np.mean((true - pred) ** 2))
def mae(true, pred):  return np.mean(np.abs(true - pred))

# --- Baseline 1: PERSISTENCE = last value in each window's history ---
pred_persist = X_test[:, -1]          # last column = most recent reading

# --- Baseline 2: MEAN = average glucose of TRAINING set (one number) ---
mean_value = y_train.mean()           # learned from TRAIN only (no peeking)
pred_mean = np.full_like(y_test, mean_value)

# --- scores on held-out subjects ---

print(f"held-out test windows: {len(y_test)}")
print(f"\nPersistence -> RMSE {rmse(y_test, pred_persist):.2f}   MAE {mae(y_test, pred_persist):.2f}")
print(f"Mean ({mean_value:.1f}) -> RMSE {rmse(y_test, pred_mean):.2f}   MAE {mae(y_test, pred_mean):.2f}")