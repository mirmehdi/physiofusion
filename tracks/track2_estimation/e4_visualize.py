"""Phase 1 EDA: plot all signals for ONE subject, aligned on a shared
time axis, over a short window — to SEE the data before building features.
"""
from pathlib import Path                                    # path handling
import pandas as pd                                         # data loading
import numpy as np                                          # math (motion magnitude)
import matplotlib.pyplot as plt                             # plotting

# --- locate one subject's folder ---
ROOT = Path(__file__).resolve().parents[2]                  # project root
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"  # dataset
SUBJECT = "001"                                             # which subject to view
folder = BIG / SUBJECT                                      # this subject's folder

# --- helper: load one E4 signal, clean its column names ---
def load_signal(name, value_col):                          # name=file prefix, value_col=data column
    df = pd.read_csv(folder / f"{name}_{SUBJECT}.csv")     # read the CSV
    df.columns = [c.strip() for c in df.columns]           # strip leading spaces from column names
    df["datetime"] = pd.to_datetime(df["datetime"])        # parse timestamps to real datetimes
    return df                                              # return tidy dataframe

# --- load the signals we care about (skip giant BVP for now) ---
hr   = load_signal("HR", "hr")                             # heart rate, 1 Hz
eda  = load_signal("EDA", "eda")                           # electrodermal activity, 4 Hz
temp = load_signal("TEMP", "temp")                         # skin temperature, 4 Hz
acc  = load_signal("ACC", "acc_x")                         # accelerometer, 32 Hz (3 axes)

# --- motion magnitude from the 3 accelerometer axes: sqrt(x^2+y^2+z^2) ---
acc["motion"] = np.sqrt(acc["acc_x"]**2 + acc["acc_y"]**2 + acc["acc_z"]**2)  # combine axes into one motion signal

# --- load glucose (Dexcom), same cleaning you already know ---
raw = pd.read_csv(folder / f"Dexcom_{SUBJECT}.csv")        # read Dexcom file
egv = raw[raw["Event Type"] == "EGV"].copy()               # keep only real glucose readings
egv["datetime"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])  # parse timestamp
egv["glucose"] = pd.to_numeric(egv["Glucose Value (mg/dL)"], errors="coerce")  # glucose to numeric
cgm = egv[["datetime", "glucose"]].sort_values("datetime")  # tidy glucose frame

# --- pick a short window to view: first 6 hours of data ---
start = cgm["datetime"].min()                              # earliest glucose reading
end = start + pd.Timedelta(hours= 10)                        # 6 hours later

def clip(df):                                              # keep only rows inside [start, end]
    return df[(df["datetime"] >= start) & (df["datetime"] <= end)]  # time-window filter

# --- plot: 5 stacked panels sharing the same x-axis (time) ---
fig, ax = plt.subplots(5, 1, figsize=(13, 10), sharex=True)  # 5 rows, shared time axis

ax[0].plot(clip(cgm)["datetime"], clip(cgm)["glucose"], color="black")  # glucose on top
ax[0].set_ylabel("Glucose\n(mg/dL)")                       # label

ax[1].plot(clip(hr)["datetime"], clip(hr)["hr"], color="crimson")  # heart rate
ax[1].set_ylabel("HR\n(bpm)")                              # label

ax[2].plot(clip(eda)["datetime"], clip(eda)["eda"], color="teal")  # EDA
ax[2].set_ylabel("EDA\n(µS)")                              # label

ax[3].plot(clip(temp)["datetime"], clip(temp)["temp"], color="darkorange")  # temperature
ax[3].set_ylabel("Temp\n(°C)")                             # label

ax[4].plot(clip(acc)["datetime"], clip(acc)["motion"], color="purple", linewidth=0.5)  # motion
ax[4].set_ylabel("Motion")                                 # label
ax[4].set_xlabel("Time")                                   # x-axis label (bottom panel only)

plt.suptitle(f"Subject {SUBJECT} — all signals, first 10 hours")  # overall title
plt.tight_layout()                                         # neat spacing
plt.show()                                                 # display (close window to end)