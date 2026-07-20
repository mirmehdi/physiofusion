"""Test HR feature extraction on one subject, aligned to glucose, gap-safe."""
from pathlib import Path                                    # paths
import pandas as pd                                         # data
from physiofusion.features import load_e4_signal, summarize_to_cgm  # our new tools

ROOT = Path(__file__).resolve().parents[2]                  # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # data
SUBJECT = "001"                                             # subject
folder = BIG / SUBJECT                                      # subject folder

# --- load glucose timestamps (the master clock) ---
raw = pd.read_csv(folder / f"Dexcom_{SUBJECT}.csv")         # read Dexcom
egv = raw[raw["Event Type"] == "EGV"].copy()               # keep glucose readings
egv["datetime"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])  # parse time
cgm_times = egv.sort_values("datetime")["datetime"].to_numpy()  # sorted glucose timestamps

# --- load HR and summarize onto the glucose grid ---
hr = load_e4_signal(folder, "HR", SUBJECT, "hr", clip_range=(30, 200))  # HR: drop <30 or >200 bpm (artifacts)          # load HR signal
hr_feats = summarize_to_cgm(hr, "hr", cgm_times)           # 5-min features aligned to glucose

# --- inspect ---
print("glucose timestamps:", len(cgm_times))               # how many slots
print("HR feature table shape:", hr_feats.shape)           # rows should match glucose count
print("\nfirst 5 rows:")                                   # peek
print(hr_feats.head())                                     # show features
print("\nhow many slots had a GAP (NaN features)?")        # gap check
print(hr_feats["hr_mean"].isna().sum(), "of", len(hr_feats))  # count missing
print("\ncoverage stats (samples per 5-min slot):")        # coverage sanity
print(hr_feats["hr_coverage"].describe())                  # typical ~300 at 1 Hz