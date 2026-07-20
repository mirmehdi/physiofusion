"""Build + save the multi-channel SEQUENCE dataset for early fusion."""
from pathlib import Path                                       # paths
from physiofusion.data.multimodal import build_sequence_dataset, save_sequence_dataset  # builders

ROOT = Path(__file__).resolve().parents[2]                    # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # raw data
OUT = ROOT / "Data" / "processed"                             # output folder

SUBJECTS = [f"{i:03d}" for i in range(1, 17) if i != 15]      # 15 subjects (drop 015)

SIGNAL_SPECS = [                                              # same signals as before
    {"name": "HR",   "col": "hr",   "clip": (30, 200)},      # heart rate, artifact-clipped
    {"name": "EDA",  "col": "eda"},                           # electrodermal
    {"name": "TEMP", "col": "temp"},                          # skin temp
    {"name": "ACC",  "col": "acc_x", "acc": True, "min_cov": 1000},  # motion magnitude
]

# the 5 channels: glucose + one mean per wristband signal (full history each)
CHANNEL_COLS = ["glucose", "hr_mean", "eda_mean", "temp_mean", "motion_mean"]  # 5 channels

print("building sequence dataset...")                         # start
X, y, groups = build_sequence_dataset(BIG, SUBJECTS, SIGNAL_SPECS, CHANNEL_COLS)  # build

print(f"\nfinal: X {X.shape}  (windows, channels, timesteps)")  # should be (~24348, 5, 24)
print(f"       y {y.shape}, {len(set(groups))} subjects")     # targets + subject count
save_sequence_dataset(X, y, groups, OUT)                      # save .npy files