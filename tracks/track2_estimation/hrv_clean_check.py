"""Compare naive vs Kubios-corrected HRV on one subject."""
from pathlib import Path                                       # paths
import pandas as pd                                            # data
from physiofusion.features import summarize_hrv_clean_to_cgm # both versions

ROOT = Path(__file__).resolve().parents[2]                    # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # data
SUBJECT = "001"                                               # subject
folder = BIG / SUBJECT                                        # folder

# --- glucose clock ---
raw = pd.read_csv(folder / f"Dexcom_{SUBJECT}.csv")           # Dexcom
egv = raw[raw["Event Type"] == "EGV"].copy()                 # glucose rows
egv["datetime"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])  # parse
cgm_times = egv.sort_values("datetime")["datetime"].to_numpy()  # timestamps

# --- IBI ---
ibi = pd.read_csv(folder / f"IBI_{SUBJECT}.csv")             # read IBI
ibi.columns = [c.strip() for c in ibi.columns]               # strip spaces
ibi["datetime"] = pd.to_datetime(ibi["datetime"])            # parse
ibi = ibi.sort_values("datetime").reset_index(drop=True)     # sort

print("running Kubios-corrected HRV (slower — artifact correction per slot)...")  # note
hrv_clean = summarize_hrv_clean_to_cgm(ibi, cgm_times)       # the corrected version

print("\n=== CORRECTED (Kubios) ===")                        # results
print("missing fraction:", hrv_clean["hrv_rmssd"].isna().mean().round(3))  # coverage
print("\nHRV ranges:")                                       # plausibility
print(hrv_clean[["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_meannn"]].describe().round(1).to_string())
print("\nartifacts corrected per slot:")                     # how much cleaning happened
print(hrv_clean["hrv_n_artifacts"].describe().round(1))      # diagnostic