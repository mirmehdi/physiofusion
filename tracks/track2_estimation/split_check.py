from pathlib import Path
import numpy as np
from physiofusion.windowing import build_dataset
from physiofusion.splits import subject_grouped_split

ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"

subjects = [f"{i:03d}" for i in range(1, 17) if i != 15]
X, y, groups = build_dataset(BIG, subjects)

# make one split: fold 0 is the test pile
train_idx, test_idx = subject_grouped_split(groups, n_splits=5, fold=0)

print("\n=== split result ===")
print(f"total windows: {len(X)}")
print(f"train windows: {len(train_idx)}   test windows: {len(test_idx)}")

# WHO is in each pile?
train_subjects = set(groups[train_idx])
test_subjects  = set(groups[test_idx])
print(f"\ntrain subjects: {sorted(train_subjects)}")
print(f"test subjects:  {sorted(test_subjects)}")

# THE CRUCIAL CHECK: is anyone in both piles?
overlap = train_subjects & test_subjects
print(f"\nsubjects in BOTH piles: {overlap}   (must be empty!)")