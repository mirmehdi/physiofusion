"""Phase 1 EDA: inspect the raw Empatica E4 signal files for ONE subject.
Just LOOK before building anything — print format, shape, first rows.
"""

from pathlib import Path 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
SUBJECT = "001"
folder = BIG / SUBJECT
folder = "Data/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/001"
# the wristband signal files (plus Dexcom for reference)
files = ["BVP", "EDA", "TEMP", "HR", "ACC", "Dexcom"]

for name in files:
    path = folder / f"{name}_{SUBJECT}.csv"
    print("=" * 60)
    print(f"{name}  ->  {path.name}")
    if not path.exists():
        print("  (file not found)")
        continue
    raw = pd.read_csv(path)
    print(f"  shape: {raw.shape}")
    print(f"  columns: {list(raw.columns)}")
    print("  first 5 rows:")
    print(raw.head().to_string())
    print()



