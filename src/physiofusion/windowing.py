"""Stage 1 signal front-end — reusable across every track and dataset.

Two tools:
  segment_by_gaps : cut one subject's readings into gap-free 'islands'
  make_windows    : slide over ONE island to build training examples

These are the reusable knives. Tracks import them; they are never
redefined inside a track file. One knife, one drawer.
"""
import numpy as np
import pandas as pd


def segment_by_gaps(cgm, gap_threshold_min=15):
    """Split a CGM series into continuous runs ('islands').

    A new island begins whenever the gap to the previous reading is
    larger than gap_threshold_min. Inside any island, every consecutive
    step is small, so it is safe to build windows there.

    cgm : DataFrame with a datetime 'ts' column, one row per reading.
    returns: the same DataFrame with an added integer 'island' column.
    """
    # Never trust file order; sort by time and take our own private copy.
    cgm = cgm.sort_values("ts").reset_index(drop=True).copy()

    # Minutes between each reading and the one before it.
    # The first row has no predecessor -> NaN, which is fine below.
    dt_min = cgm["ts"].diff().dt.total_seconds().div(60)

    # True exactly on rows that come AFTER a big gap (they start a new island).
    # (NaN > threshold) is False, so the first row quietly joins island 0.
    is_new_island = dt_min > gap_threshold_min

    # Running total of the True/False column: ticks up by 1 at each gap,
    # stays flat in between. That running total IS the island number.
    cgm["island"] = is_new_island.cumsum()
    return cgm


def make_windows(island, history=24, horizon=6, step=1):
    """Turn ONE continuous island into (history -> future glucose) examples.

    history : past readings the model sees   (24 = 2 hours at 5-min spacing)
    horizon : readings ahead we predict       (6  = 30 minutes)
    step    : how far the frame slides each time (1 = 5 minutes)

    IMPORTANT: pass a SINGLE island here, never a whole subject. Looping
    over islands/subjects happens one level up, on purpose, so this stays
    a small, testable brick.

    returns: X of shape (n_windows, history), y of shape (n_windows,)
    """
    g = island["glucose"].to_numpy()   # glucose column as a plain array
    n = len(g)
    X, y = [], []

    start = 0
    # The frame touches readings from 'start' up to (start+history+horizon-1).
    # That last index must stay inside the island, i.e. < n. Rearranged:
    #     start + history + horizon <= n
    while start + history + horizon <= n:
        X.append(g[start : start + history])          # the question: past history
        y.append(g[start + history + horizon - 1])    # the answer: one point ahead
        start += step                                 # slide the frame forward

    # Stack into arrays. If the island was too short for even one window,
    # return correctly-shaped empties so downstream code never crashes.
    if X:
        return np.array(X), np.array(y)
    return np.empty((0, history)), np.empty((0,))

def load_subject_cgm(big_dir, subject):
    """Load and clean ONE subject's Dexcom CGM into a tidy DataFrame.

    Does the boring, repeated chores in one place:
      - read the subject's Dexcom CSV
      - keep only real glucose readings (Event Type == 'EGV')
      - parse the timestamp text into a real datetime
      - keep just two columns: 'ts' and 'glucose', sorted by time

    big_dir : Path to the BIG IDEAs dataset folder
    subject : subject id as a string, e.g. '002'
    returns : DataFrame with columns ['ts', 'glucose'], one row per reading
    """
    path = big_dir / subject / f"Dexcom_{subject}.csv"

    raw = pd.read_csv(path)

    # The first ~12 rows are patient metadata; real readings are 'EGV'.
    egv = raw[raw["Event Type"] == "EGV"].copy()

    # Turn the timestamp text into a real datetime so we can do time math.
    egv["ts"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])

    # Keep only what we need, rename to a short column, sort by time.
    cgm = (
        egv[["ts", "Glucose Value (mg/dL)"]]
        .rename(columns={"Glucose Value (mg/dL)": "glucose"})
        .sort_values("ts")
        .reset_index(drop=True)
    )

    # Make sure glucose is numeric; anything unparseable becomes NaN.
    cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")

    return cgm

def windows_for_subject(big_dir, subject, history=24, horizon=6, step=1):
    """All windows for ONE subject, tagged with their owner.

    Loads the subject, segments into islands, windows EACH island
    separately (never across a gap), and stacks the results — attaching
    the subject id to every window so we never lose track of whose it is.

    returns: X (n, history), y (n,), groups (n,) of the subject id
    """
    cgm = load_subject_cgm(big_dir, subject)      # load + clean
    seg = segment_by_gaps(cgm)                    # cut into islands

    X_parts, y_parts = [], []

    # window each island on its own, so no window ever spans a gap
    for island_id, island in seg.groupby("island"):
        Xi, yi = make_windows(island, history, horizon, step)
        if len(Xi):                               # skip islands too short to window
            X_parts.append(Xi)
            y_parts.append(yi)

    # if the subject produced no windows at all, return empties
    if not X_parts:
        return (np.empty((0, history)), np.empty((0,)), np.empty((0,), dtype=object))

    X = np.concatenate(X_parts)                   # stack all islands' questions
    y = np.concatenate(y_parts)                   # stack all islands' answers

    # the owner sticker: the subject id, one per window
    groups = np.full(len(X), subject, dtype=object)

    return X, y, groups

def build_dataset(big_dir, subjects, history=24, horizon=6, step=1):
    """Build the FULL training set from many subjects.

    Calls windows_for_subject on each subject and stacks everyone into
    one big X, y, groups. The 'groups' array remembers whose each window
    is — which is what the leakage-proof splitter will use later.

    subjects : list of subject ids to include, e.g. ['001','002',...]
               (leave 015 out of this list — it's the dropped subject)

    returns : X (N, history), y (N,), groups (N,)
    """
    X_parts, y_parts, g_parts = [], [], []

    for subject in subjects:
        Xs, ys, gs = windows_for_subject(big_dir, subject, history, horizon, step)
        print(f"  subject {subject}: {len(Xs):>5} windows")
        if len(Xs):
            X_parts.append(Xs)
            y_parts.append(ys)
            g_parts.append(gs)

    # stack every subject's pile into one
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)

    return X, y, groups