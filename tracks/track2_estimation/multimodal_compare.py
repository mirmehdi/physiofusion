"""Multimodal result: does glucose + wristband beat glucose-only (13.90)?
Loads the saved Parquet, runs 5-fold subject-grouped CV, compares feature sets.
"""
from pathlib import Path                                     # paths
import numpy as np                                          # math
import pandas as pd                                         # load parquet
from physiofusion.splits import subject_grouped_split       # leakage-proof split
from sklearn.linear_model import LinearRegression           # our champion model
from sklearn.pipeline import make_pipeline                  # scaler + model
from sklearn.preprocessing import StandardScaler            # feature scaling

ROOT = Path(__file__).resolve().parents[2]                 # project root
OUT = ROOT / "Data" / "processed" / "multimodal.parquet"   # the saved dataset

# --- load the dataset (instant now — no rebuilding) ---
df = pd.read_parquet(OUT)                                   # read parquet
groups = df["subject"].to_numpy()                          # subject tags
y = df["y"].to_numpy()                                     # target (glucose +30 min)

# --- define two feature sets to COMPARE ---
glucose_cols = [c for c in df.columns if c.startswith("glucose_lag")]  # the 24 glucose lags only
wrist_cols   = [c for c in df.columns if c not in glucose_cols + ["y", "subject"]]  # wristband features
all_cols     = glucose_cols + wrist_cols                   # glucose + wristband

feature_sets = {                                           # the experiments to run
    "glucose-only":       glucose_cols,                    # 24 glucose lags
    "glucose + wristband": all_cols,                       # + 16 wristband features
}

def rmse(t, p): return np.sqrt(np.mean((t - p) ** 2))      # RMSE helper
def mae(t, p):  return np.mean(np.abs(t - p))              # MAE helper

# --- run 5-fold subject-grouped CV for each feature set ---
for set_name, cols in feature_sets.items():                # each experiment
    X = df[cols].to_numpy()                                # features for this set
    rmses, maes = [], []                                   # per-fold scores
    for fold in range(5):                                  # 5 folds
        tr, te = subject_grouped_split(groups, n_splits=5, fold=fold)  # split by subject
        model = make_pipeline(StandardScaler(), LinearRegression())    # scaler + linear
        model.fit(X[tr], y[tr])                            # train on training subjects
        pred = model.predict(X[te])                        # predict held-out subjects
        rmses.append(rmse(y[te], pred))                    # record RMSE
        maes.append(mae(y[te], pred))                      # record MAE
    print(f"{set_name:22s} RMSE {np.mean(rmses):.2f} ± {np.std(rmses):.2f}   "
          f"MAE {np.mean(maes):.2f} ± {np.std(maes):.2f}")  # report mean ± std