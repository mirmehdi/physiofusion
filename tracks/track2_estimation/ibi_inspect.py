"""Inspect the IBI (inter-beat interval) signal — the source for HRV features."""
from pathlib import Path                                     # paths
import pandas as pd                                          # data

ROOT = Path(__file__).resolve().parents[2]                  # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # data
SUBJECT = "001"                                             # start with one subject

ibi = pd.read_csv(BIG / SUBJECT / f"IBI_{SUBJECT}.csv")     # read the IBI file
ibi.columns = [c.strip() for c in ibi.columns]              # strip leading spaces from column names

print("shape:", ibi.shape)                                  # how many beats recorded
print("columns:", list(ibi.columns))                        # what columns exist
print("\nfirst 10 rows:")                                   # peek at the format
print(ibi.head(10).to_string())                             # show them
print("\ndtypes:")                                          # data types
print(ibi.dtypes)                                           # check what's numeric
print("\nvalue stats (the interval column):")               # sanity-check the intervals
print(ibi.describe().to_string())                           # typical IBI ~0.6-1.2 sec (50-100 bpm)