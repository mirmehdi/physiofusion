"""Throwaway check: does windowing work on subject 002? (saves nothing)"""
from pathlib import Path
import pandas as pd

# borrow BOTH knives from the library
from physiofusion.windowing import segment_by_gaps, make_windows

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
SUBJECT = "002"

# --- load + clean this subject's CGM (steps you already know) ---
raw = pd.read_csv(BIG / SUBJECT / f"Dexcom_{SUBJECT}.csv")
egv = raw[raw["Event Type"] == "EGV"].copy()
egv["ts"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])
cgm = (
    egv[["ts", "Glucose Value (mg/dL)"]]
    .rename(columns={"Glucose Value (mg/dL)": "glucose"})
    .sort_values("ts")
    .reset_index(drop=True)
)
cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")

# --- segment, then window just island 0 (the part before the 14-hour gap) ---
seg = segment_by_gaps(cgm, gap_threshold_min=15)
island0 = seg[seg["island"] == 0]

X, y = make_windows(island0, history=24, horizon=6, step=1)

print(f"island 0: {len(island0)} readings -> {len(X)} windows")
print(f"X shape: {X.shape}   (n_windows, history)")
print(f"y shape: {y.shape}")
print("\nfirst window history (24 glucose values):")
print(X[0])
print(f"its answer (glucose 30 min after history ends): {y[0]}")