from pathlib import Path
from physiofusion.windowing import build_dataset

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"

# all 16 folders exist, but we DROP 015 (75% complete, 22-hour gap)
subjects = [f"{i:03d}" for i in range(1, 17) if i != 15]
print("subjects:", subjects)

X, y, groups = build_dataset(BIG, subjects)

print("\n=== full dataset ===")
print("X:", X.shape, " y:", y.shape, " groups:", groups.shape)
print("subjects included:", sorted(set(groups)))
print("windows per subject:")
import collections
for s, c in sorted(collections.Counter(groups).items()):
    print(f"  {s}: {c}")