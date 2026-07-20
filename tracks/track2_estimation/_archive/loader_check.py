from pathlib import Path
from physiofusion.windowing import load_subject_cgm

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"

cgm = load_subject_cgm(BIG, "002")
print(cgm.shape)     # expect (2119, 2)
print(cgm.head())
