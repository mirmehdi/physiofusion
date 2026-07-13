"""Compare models across ALL 5 folds (each subject group is test once).

One fold can mislead. Looping all folds and reporting mean ± std across
folds gives an honest estimate of how each model generalizes to new people.
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

def rmse(true, pred): return np.sqrt(np.mean((true - pred) ** 2))
def mae(true, pred):  return np.mean(np.abs(true - pred))

# fresh model each fold (a builder, so weights don't carry over between folds)
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
model_names = list(make_models().keys()) + ["Persistence"]

# collect per-fold RMSE for every model: {name: [rmse_fold0, rmse_fold1, ...]}
rmse_scores = {name: [] for name in model_names}
mae_scores  = {name: [] for name in model_names}

# ---- loop every fold: each subject group serves as test exactly once ----
for fold in range(N_SPLITS):
    tr, te = subject_grouped_split(groups, n_splits=N_SPLITS, fold=fold)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # persistence reference for this fold
    p = X_te[:, -1]
    rmse_scores["Persistence"].append(rmse(y_te, p))
    mae_scores["Persistence"].append(mae(y_te, p))

    # train + score each model on this fold
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse_scores[name].append(rmse(y_te, pred))
        mae_scores[name].append(mae(y_te, pred))

    print(f"  fold {fold} done")

# ---- report mean ± std across the 5 folds, best mean-RMSE first ----
print(f"\n{'Model':<15} {'RMSE (mean±std)':>20} {'MAE (mean±std)':>20}")
print("-" * 57)
order = sorted(model_names, key=lambda n: np.mean(rmse_scores[n]))
for name in order:
    r = np.array(rmse_scores[name]); m = np.array(mae_scores[name])
    print(f"{name:<15} {r.mean():>8.2f} ± {r.std():>5.2f}     {m.mean():>8.2f} ± {m.std():>5.2f}")