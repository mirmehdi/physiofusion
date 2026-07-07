"""
Segment one subject's CGM into gap-free 'islands'.

WHY THIS EXISTS:
Some subjects have big holes in their CGM (subject 002 has a ~14-hour gap
where the sensor was off). If we later slide a training window across such a
hole, the model would treat two readings 14 hours apart as if they were only
5 minutes apart — a lie that corrupts everything. So before any windowing, we
cut each subject's data at the holes into continuous pieces we call "islands".
Windows will only ever be built INSIDE one island, never across a cut.

Run from the project root:
    python tracks/track2_estimation/segment_check.py
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------------------------
SUBJECT = "002"          # start with 002 because we KNOW it has a 14-hour gap
GAP_THRESHOLD_MIN = 15   # any interval bigger than this = a real break = a cut

# Find the data folder relative to THIS file, so the script works no matter
# what folder you run it from. parents[2] climbs three levels up:
#   this file -> track2_estimation -> tracks -> A_glucose  (the project root)
ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
dexcom_path = BIG / SUBJECT / f"Dexcom_{SUBJECT}.csv"


# ---------------------------------------------------------------------------
# 1. The segmenter — the one important idea in this whole file
# ---------------------------------------------------------------------------
from physiofusion.windowing import segment_by_gaps
# ---------------------------------------------------------------------------
# 2. Load and clean this subject's CGM (same steps as your eda.py)
# ---------------------------------------------------------------------------
raw = pd.read_csv(dexcom_path)

# The first ~12 rows of a Dexcom file are patient metadata, not readings.
# Real glucose readings are the rows where Event Type == "EGV"
# (EGV = "Estimated Glucose Value"). Keep only those.
egv = raw[raw["Event Type"] == "EGV"].copy()

# Turn the text timestamp into a real datetime so we can do time math on it.
egv["ts"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])

# Keep just the two columns we care about, rename glucose to something short,
# sort by time, and reset the row numbers to 0,1,2,...
cgm = (
    egv[["ts", "Glucose Value (mg/dL)"]]
    .rename(columns={"Glucose Value (mg/dL)": "glucose"})
    .sort_values("ts")
    .reset_index(drop=True)
)
# Make sure glucose is a number; anything unparseable becomes NaN.
cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")


# ---------------------------------------------------------------------------
# 3. Run the segmenter and look at the result
# ---------------------------------------------------------------------------
seg = segment_by_gaps(cgm, gap_threshold_min=GAP_THRESHOLD_MIN)

print(f"\nSubject {SUBJECT}: {len(seg)} total readings")

# How many readings landed in each island?
print("\nreadings per island:")
print(seg["island"].value_counts().sort_index())

# Describe each island: how many readings, and its start/end time + duration.
print("\nisland details:")
for island_id, island in seg.groupby("island"):
    span = island["ts"].max() - island["ts"].min()
    print(f"  island {island_id}: {len(island):>4} readings | "
          f"{island['ts'].min()} -> {island['ts'].max()}  ({span})")

# Sanity check: the islands must add back up to the total (nothing lost).
print(f"\ncheck: islands sum to {seg['island'].value_counts().sum()} "
      f"(should equal {len(seg)})")