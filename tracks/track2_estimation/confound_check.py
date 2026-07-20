"""THE CONFOUND EXPERIMENT.

Question: when a wristband appears to 'predict glucose', what is it actually
keying on — physiology, or just the time of day?

We compare feature sets on IDENTICAL windows and folds, so differences are
purely due to the inputs:
  1. time-only      -> pure circadian confound (no physiology at all)
  2. wristband-only -> can physiology beat the clock?
  3. wristband+time -> does physiology add anything on top of time?
  4. glucose-only   -> reference (what CGM history alone achieves)
Also a mean baseline for the floor.
"""
from pathlib import Path                                     # paths
import numpy as np                                           # math
import pandas as pd                                          # load parquet
from physiofusion.splits import subject_grouped_split        # leakage-proof split
from sklearn.linear_model import LinearRegression            # our champion model
from sklearn.ensemble import GradientBoostingRegressor       # nonlinear check
from sklearn.pipeline import make_pipeline                   # scaler + model
from sklearn.preprocessing import StandardScaler             # feature scaling

ROOT = Path(__file__).resolve().parents[2]                   # project root
df = pd.read_parquet(ROOT / "Data" / "processed" / "multimodal.parquet")  # load dataset

groups = df["subject"].to_numpy()                            # subject tags (for splitting)
y = df["y"].to_numpy()                                       # target: glucose +30 min

# --- define the feature sets to compare ---
glucose_cols = [c for c in df.columns if c.startswith("glucose_lag")]  # 24 CGM lags
time_cols    = ["hour_sin", "hour_cos"]                      # ONLY the clock — no physiology
wrist_cols   = [c for c in df.columns                        # the 16 wristband features
                if c not in glucose_cols + time_cols + ["y", "subject"]]

feature_sets = {                                             # each experiment
    "1. time-only":       time_cols,                         # pure confound baseline
    "2. wristband-only":  wrist_cols,                        # physiology alone
    "3. wristband+time":  wrist_cols + time_cols,            # both
    "4. glucose-only":    glucose_cols,                      # reference (CGM history)
}

def rmse(t, p): return np.sqrt(np.mean((t - p) ** 2))        # RMSE helper
def mae(t, p):  return np.mean(np.abs(t - p))                # MAE helper

# --- mean baseline: predict the training mean for everything (the floor) ---
mean_rmses = []                                              # per-fold scores
for fold in range(5):                                        # 5 folds
    tr, te = subject_grouped_split(groups, n_splits=5, fold=fold)  # split by subject
    pred = np.full(len(te), y[tr].mean())                    # predict train mean (knows nothing)
    mean_rmses.append(rmse(y[te], pred))                     # score it
print(f"{'0. mean baseline':22s} RMSE {np.mean(mean_rmses):5.2f} ± {np.std(mean_rmses):.2f}")

# --- each feature set, with BOTH a linear and a nonlinear model ---
for model_name, make_model in [                              # test two model families
    ("linear", lambda: make_pipeline(StandardScaler(), LinearRegression())),  # linear
    ("gboost", lambda: GradientBoostingRegressor(random_state=0)),            # nonlinear
]:
    print(f"\n--- {model_name} ---")                         # header per model type
    for set_name, cols in feature_sets.items():              # each feature set
        X = df[cols].to_numpy()                              # this set's features
        rmses, maes = [], []                                 # per-fold scores
        for fold in range(5):                                # 5-fold subject-grouped CV
            tr, te = subject_grouped_split(groups, n_splits=5, fold=fold)  # split
            model = make_model()                             # fresh model each fold
            model.fit(X[tr], y[tr])                          # train on training subjects
            pred = model.predict(X[te])                      # predict held-out subjects
            rmses.append(rmse(y[te], pred))                  # record RMSE
            maes.append(mae(y[te], pred))                    # record MAE
        print(f"{set_name:22s} RMSE {np.mean(rmses):5.2f} ± {np.std(rmses):.2f}   "
              f"MAE {np.mean(maes):5.2f}")                   # report mean ± std