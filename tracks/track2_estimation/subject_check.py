from pathlib import Path
from physiofusion.windowing import windows_for_subject

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"

X, y, groups = windows_for_subject(BIG, "002")
print("X:", X.shape, " y:", y.shape, " groups:", groups.shape)
print("unique owners in groups:", set(groups))
print("first owner tag:", groups[0], " last owner tag:", groups[-1])