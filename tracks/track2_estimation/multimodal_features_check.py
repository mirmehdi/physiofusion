"""Build the full multimodal feature table for ONE subject:
glucose + HR + EDA + TEMP + motion, all summarized per 5-min slot, gap-safe.
"""
from pathlib import Path                                     # paths
import numpy as np                                           # math (motion magnitude)
import pandas as pd                                          # data
from physiofusion.features import load_e4_signal, summarize_to_cgm  # our tools

ROOT = Path(__file__).resolve().parents[2]                   # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # data
SUBJECT = "001"                                              # subject
folder = BIG / SUBJECT                                       # subject folder

# ---------------------------------------------------------------------------
# 1. Glucose = master clock (the timestamps everything aligns to)
# ---------------------------------------------------------------------------
raw = pd.read_csv(folder / f"Dexcom_{SUBJECT}.csv")          # read Dexcom
egv = raw[raw["Event Type"] == "EGV"].copy()                # keep glucose readings
egv["datetime"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])  # parse time
egv["glucose"] = pd.to_numeric(egv["Glucose Value (mg/dL)"], errors="coerce")  # numeric glucose
cgm = egv[["datetime", "glucose"]].sort_values("datetime").reset_index(drop=True)  # tidy glucose
cgm_times = cgm["datetime"].to_numpy()                      # the master timestamp array

# ---------------------------------------------------------------------------
# 2. HR, EDA, TEMP -> load + summarize (same pattern for each)
# ---------------------------------------------------------------------------
hr   = load_e4_signal(folder, "HR",   SUBJECT, "hr",   clip_range=(30, 200))  # HR, artifact-clipped
eda  = load_e4_signal(folder, "EDA",  SUBJECT, "eda")       # EDA (electrodermal)
temp = load_e4_signal(folder, "TEMP", SUBJECT, "temp")      # skin temperature

hr_feats   = summarize_to_cgm(hr,   "hr",   cgm_times)       # HR features per 5-min slot
eda_feats  = summarize_to_cgm(eda,  "eda",  cgm_times)       # EDA features
temp_feats = summarize_to_cgm(temp, "temp", cgm_times)       # TEMP features

# ---------------------------------------------------------------------------
# 3. Motion: combine 3 accelerometer axes into ONE magnitude, then summarize.
#    load_e4_signal only keeps one value column, so we load ACC manually here.
# ---------------------------------------------------------------------------
acc = pd.read_csv(folder / f"ACC_{SUBJECT}.csv")            # read accelerometer (3 axes)
acc.columns = [c.strip() for c in acc.columns]             # strip spaces from column names
acc["datetime"] = pd.to_datetime(acc["datetime"])          # parse timestamps
acc["motion"] = np.sqrt(acc["acc_x"]**2                     # combine axes into one motion signal
                        + acc["acc_y"]**2                   #   sqrt(x^2 + y^2 + z^2)
                        + acc["acc_z"]**2)                  #   = overall movement magnitude
acc = acc[["datetime", "motion"]].dropna().sort_values("datetime").reset_index(drop=True)  # tidy
# ACC is 32 Hz -> ~9600 samples per 5-min slot; require more coverage before trusting a slot
motion_feats = summarize_to_cgm(acc, "motion", cgm_times, min_coverage=1000)  # motion features

# ---------------------------------------------------------------------------
# 4. Stitch everything into ONE table: glucose + all signal features
# ---------------------------------------------------------------------------
table = pd.concat([                                         # glue columns side by side
    cgm[["datetime", "glucose"]],                          # timestamp + target
    hr_feats.drop(columns="hr_coverage"),                  # HR features (drop coverage col for clarity)
    eda_feats.drop(columns="eda_coverage"),                # EDA features
    temp_feats.drop(columns="temp_coverage"),              # TEMP features
    motion_feats.drop(columns="motion_coverage"),          # motion features
], axis=1)                                                 # axis=1 = join as columns

# ---------------------------------------------------------------------------
# 5. Inspect the result
# ---------------------------------------------------------------------------
print("full table shape:", table.shape)                    # rows = glucose count, cols = all features
print("\ncolumns:", list(table.columns))                   # see every feature name
print("\nfirst 5 rows:")                                   # peek
print(table.head().to_string())                            # show the multimodal table

# how complete is each signal? (fraction of slots with real data)
print("\nmissing (NaN) fraction per feature:")             # gap summary across signals
print(table.isna().mean().round(3))                        # 0 = never missing, 1 = always missing