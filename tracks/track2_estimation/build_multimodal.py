"""Build + save the multimodal dataset for all subjects."""
from pathlib import Path                                    # paths
from physiofusion.data.multimodal import build_multimodal_dataset, save_dataset  # builder + saver

ROOT = Path(__file__).resolve().parents[2]                 # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # data
OUT = ROOT / "Data" / "processed"                          # output folder

SUBJECTS = [f"{i:03d}" for i in range(1, 17) if i != 15]   # 15 subjects (drop 015)

# the signals to include (matches what you tested)
SIGNAL_SPECS = [                                           # each signal's load recipe
    {"name": "HR",   "col": "hr",   "clip": (30, 200)},   # heart rate, artifact-clipped
    {"name": "EDA",  "col": "eda"},                        # electrodermal
    {"name": "TEMP", "col": "temp"},                       # skin temp
    {"name": "ACC",  "col": "acc_x", "acc": True, "min_cov": 1000},  # motion (3-axis magnitude)
]

print("building multimodal dataset...")                    # start
X, y, groups, feature_names = build_multimodal_dataset(BIG, SUBJECTS, SIGNAL_SPECS)  # build it

print(f"\nfinal: X {X.shape}, y {y.shape}, {len(set(groups))} subjects")  # summary
print("features:", feature_names)                          # the column names
save_dataset(X, y, groups, feature_names, OUT)             # save to Parquet