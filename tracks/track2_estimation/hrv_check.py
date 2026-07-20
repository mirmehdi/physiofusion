"""Check HRV extraction on one subject — coverage and plausibility."""
from pathlib import Path                                      # paths
import pandas as pd                                           # data
from physiofusion.features import summarize_hrv_to_cgm        # our new extractor

ROOT = Path(__file__).resolve().parents[2]                   # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # data
SUBJECT = "001"                                              # subject
folder = BIG / SUBJECT                                       # subject folder

# --- glucose timestamps (the master clock) ---
raw = pd.read_csv(folder / f"Dexcom_{SUBJECT}.csv")          # read Dexcom
egv = raw[raw["Event Type"] == "EGV"].copy()                # keep glucose readings
egv["datetime"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])  # parse time
cgm_times = egv.sort_values("datetime")["datetime"].to_numpy()  # sorted timestamps

# --- load IBI ---
ibi = pd.read_csv(folder / f"IBI_{SUBJECT}.csv")            # read IBI file
ibi.columns = [c.strip() for c in ibi.columns]              # strip column spaces
ibi["datetime"] = pd.to_datetime(ibi["datetime"])           # parse timestamps
ibi = ibi.sort_values("datetime").reset_index(drop=True)    # sort by time

# --- extract HRV features ---
hrv = summarize_hrv_to_cgm(ibi, cgm_times)                  # per-slot HRV

print("glucose slots:", len(cgm_times))                     # how many slots
print("HRV table shape:", hrv.shape)                        # should match
print("\nfirst 5 rows:")                                    # peek
print(hrv.head().to_string())                               # show features

print("\nmissing fraction per feature:")                    # THE key number
print(hrv.isna().mean().round(3))                           # how much HRV is unusable

print("\nbeats per slot (diagnostic):")                     # coverage check
print(hrv["hrv_n_beats"].describe())                        # expect ~100 at 62bpm if all detected

print("\nHRV value ranges (plausibility):")                 # sanity check
print(hrv[["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_meannn"]].describe().round(1).to_string())