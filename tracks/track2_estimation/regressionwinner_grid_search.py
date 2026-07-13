"""Compare models across ALL 5 folds -> mean ± std per model.

Each fold holds out ~3 different subjects. Averaging over folds gives an
honest estimate; the spread (std) shows how much the result depends on
WHICH subjects are held out.
"""
from pathlib import Path
import numpy as np

from physiofusion.windowing import build_dataset
from physiofusion.splits import subject_grouped_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
SUBJECTS = [f"{i:03d}" for i in range(1, 17) if i != 15]

X, y, groups = build_dataset(BIG, SUBJECTS)

def rmse(t, p): return np.sqrt(np.mean((t - p) ** 2))
def mae(t, p):  return np.mean(np.abs(t - p))

# fresh model each fold (a fitted model can't be reused) -> use factories
def make_models():
    return {
        "Linear":        make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge":         make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso":         make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
        "ElasticNet":    make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5)),
        "RandomForest":  RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
        "GradientBoost": GradientBoostingRegressor(random_state=0),
    }

N_SPLITS = 5
# collect RMSE per model across folds: {model_name: [rmse_fold0, rmse_fold1, ...]}
scores = {name: [] for name in make_models()}
scores["Persistence"] = []

for fold in range(N_SPLITS):
    tr, te = subject_grouped_split(groups, n_splits=N_SPLITS, fold=fold)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # persistence reference for this fold
    scores["Persistence"].append(rmse(y_te, X_te[:, -1]))

    # each model: fresh, fit on this fold's train, score its test
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        scores[name].append(rmse(y_te, model.predict(X_te)))

    print(f"  fold {fold} done")

# ---------------------------------------------------------------------------
# Report: mean ± std of RMSE across folds, best mean first
# ---------------------------------------------------------------------------
print(f"\n{'Model':<15} {'RMSE mean':>10} {'± std':>8}")
print("-" * 35)
summary = {name: (np.mean(v), np.std(v)) for name, v in scores.items()}
for name, (m, s) in sorted(summary.items(), key=lambda kv: kv[1][0]):
    print(f"{name:<15} {m:>10.2f} {s:>8.2f}")