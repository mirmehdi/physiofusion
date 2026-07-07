"""Track 2 EDA — CGM (Dexcom) label exploration for the BIG IDEAs Lab dataset.

The CGM reading is our prediction target (y). Before any modeling we must
understand it per subject: how much data, its value range, and where it gaps.

Step 1: inspect one subject in detail (set DETAIL_SUBJECT, INSPECT_ONE=True).
Step 2: build a cohort table across ALL subjects.

Run from the project root:
    python tracks/track2_estimation/eda.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- locate the data relative to this file (no fragile absolute paths) ---
ROOT = Path(__file__).resolve().parents[2]          # .../A_glucose
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"

INSPECT_ONE = False       # True -> also print the detailed Step-1 view + plot
DETAIL_SUBJECT = "001"


def load_cgm(subject: str) -> pd.DataFrame:
    """Return a clean CGM series (ts, glucose) for one subject."""
    raw = pd.read_csv(BIG / subject / f"Dexcom_{subject}.csv")
    egv = raw[raw["Event Type"] == "EGV"].copy()          # drop metadata/alert rows
    egv["ts"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])
    cgm = (
        egv[["ts", "Glucose Value (mg/dL)"]]
        .rename(columns={"Glucose Value (mg/dL)": "glucose"})
        .sort_values("ts")
        .reset_index(drop=True)
    )
    cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")
    return cgm


def cgm_stats(subject: str) -> dict:
    """One row of summary stats for the cohort table."""
    cgm = load_cgm(subject)
    span = cgm["ts"].max() - cgm["ts"].min()
    dt_min = cgm["ts"].diff().dt.total_seconds().div(60)  # gaps between readings, minutes
    expected = span.total_seconds() / 60 / 5              # if perfectly sampled every 5 min
    return {
        "subject": subject,
        "days": span.days,
        "readings": len(cgm),
        "glucose_mean": round(cgm["glucose"].mean()),
        "glucose_std": round(cgm["glucose"].std()),
        "glucose_min": round(cgm["glucose"].min()),
        "glucose_max": round(cgm["glucose"].max()),
        "biggest_gap_min": round(dt_min.max()),
        "completeness": round(len(cgm) / expected, 3) if expected else float("nan"),
    }


def find_subjects() -> list[str]:
    """All subject folders that contain a Dexcom file (e.g. '001'..'016')."""
    subs = [p.name for p in BIG.iterdir()
            if p.is_dir() and (p / f"Dexcom_{p.name}.csv").exists()]
    return sorted(subs)


# ---------------------------------------------------------------------------
# Step 1 (optional): detailed look at one subject
# ---------------------------------------------------------------------------
if INSPECT_ONE:
    cgm = load_cgm(DETAIL_SUBJECT)
    print(f"\n=== detail: subject {DETAIL_SUBJECT} ===")
    print(cgm.describe())
    dt = cgm["ts"].diff().dt.total_seconds().div(60)
    print("interval (min) value counts:\n", dt.round(1).value_counts().head(10))
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(cgm["ts"], cgm["glucose"], lw=0.6)
    ax[0].set(title=f"Subject {DETAIL_SUBJECT} CGM", ylabel="glucose mg/dL")
    ax[1].hist(cgm["glucose"].dropna(), bins=60)
    ax[1].set(xlabel="glucose mg/dL", ylabel="count")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Step 2: cohort table across ALL subjects
# ---------------------------------------------------------------------------
rows = []
for sub in find_subjects():
    try:
        rows.append(cgm_stats(sub))
    except Exception as e:                                 # never let one bad file kill the loop
        print(f"[warn] subject {sub} failed: {e}")

table = pd.DataFrame(rows).set_index("subject")
pd.set_option("display.width", 120, "display.max_columns", None)
print("\n=== CGM cohort summary (all subjects) ===")
print(table)

# save it so later steps / the writeup can reuse it (small CSV, lives next to this script)
out = Path(__file__).with_name("cgm_cohort_summary.csv")
table.to_csv(out)
print(f"\nsaved -> {out.relative_to(ROOT)}")
