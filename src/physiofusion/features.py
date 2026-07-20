"""Extract per-5-min wristband features, aligned to CGM timestamps, gap-safe.

Reusable across signals and datasets — the library's signal front-end.
"""
import numpy as np                                          # math
import pandas as pd                                         # data handling


def load_e4_signal(folder, name, subject, value_col,        # load one E4 signal file
                   clip_range=None):                        # optional (low, high) plausibility clip
    """Load one Empatica E4 signal into a tidy (datetime, value) frame.

    clip_range: if given, drop readings outside (low, high) as artifacts,
    e.g. (30, 200) for heart rate.
    """
    df = pd.read_csv(folder / f"{name}_{subject}.csv")      # read the CSV
    df.columns = [c.strip() for c in df.columns]            # strip leading spaces in column names
    df["datetime"] = pd.to_datetime(df["datetime"])         # parse timestamps to datetimes
    df = df[["datetime", value_col]].dropna()               # keep time + value, drop empty rows

    if clip_range is not None:                              # if a plausibility range was given
        low, high = clip_range                              # unpack the bounds
        mask = (df[value_col] >= low) & (df[value_col] <= high)  # keep only in-range values
        df = df[mask]                                       # drop implausible readings (artifacts)

    return df.sort_values("datetime").reset_index(drop=True)  # sort by time, clean index


def summarize_to_cgm(signal_df, value_col, cgm_times,       # align+summarize a signal to CGM grid
                     window_min=5, min_coverage=50):        # 5-min slots; need >=50 samples or NaN
    """For each CGM timestamp t, summarize signal values in (t-5min, t].

    Returns a DataFrame aligned to cgm_times with mean/std/min/max columns.
    If a slot has fewer than min_coverage samples (a gap), its features are NaN
    — we never fabricate data across gaps.
    """
    vals = signal_df[value_col].to_numpy()                  # signal values as array
    times = signal_df["datetime"].to_numpy()                # signal timestamps as array
    window = np.timedelta64(window_min, "m")                # the 5-minute window length

    means, stds, mins, maxs, covs = [], [], [], [], []      # feature accumulators

    for t in cgm_times:                                     # for each glucose timestamp
        t = np.datetime64(t)                                # ensure numpy datetime
        in_slot = (times > t - window) & (times <= t)       # the 5-min slot before t
        slot_vals = vals[in_slot]                           # signal values in that slot

        n = len(slot_vals)                                  # how many samples we found
        covs.append(n)                                      # record coverage (gap indicator)

        if n < min_coverage:                                # GAP: too few samples -> don't fabricate
            means.append(np.nan); stds.append(np.nan)       # mark all features missing
            mins.append(np.nan);  maxs.append(np.nan)       #
        else:                                               # enough data -> summarize
            means.append(slot_vals.mean())                  # average level
            stds.append(slot_vals.std())                    # variability
            mins.append(slot_vals.min())                    # minimum
            maxs.append(slot_vals.max())                    # maximum

    prefix = value_col                                      # name features after the signal
    return pd.DataFrame({                                   # assemble feature table
        f"{prefix}_mean": means,                            # mean feature
        f"{prefix}_std":  stds,                             # std feature
        f"{prefix}_min":  mins,                             # min feature
        f"{prefix}_max":  maxs,                             # max feature
        f"{prefix}_coverage": covs,                         # sample count (diagnostic)
    })


def build_subject_table(folder, subject, signal_specs, min_coverage=50):  # one subject's multimodal table
    """Build the aligned feature table for ONE subject: glucose + all signals.

    signal_specs: list of dicts describing each signal to load, e.g.
      {"name":"HR","col":"hr","clip":(30,200)}
      {"name":"ACC","col":"acc_x","acc":True,"min_cov":1000}
    Returns a DataFrame: datetime, glucose, then <sig>_mean/std/min/max per signal.
    Rows where any signal is missing are kept as NaN here (dropped later).
    """
    # --- glucose = master clock ---
    raw = pd.read_csv(folder / f"Dexcom_{subject}.csv")     # read Dexcom
    egv = raw[raw["Event Type"] == "EGV"].copy()            # keep glucose readings
    egv["datetime"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"])  # parse time
    egv["glucose"] = pd.to_numeric(egv["Glucose Value (mg/dL)"], errors="coerce")  # numeric glucose
    cgm = egv[["datetime", "glucose"]].sort_values("datetime").reset_index(drop=True)  # tidy
    cgm_times = cgm["datetime"].to_numpy()                  # master timestamps

    # --- time-of-day features (for the confound baseline) ---
    # Encode hour cyclically: hour 23 and hour 0 are adjacent in reality, but
    # numerically far apart. sin/cos maps them onto a circle so they're close.
    hour = cgm["datetime"].dt.hour + cgm["datetime"].dt.minute / 60  # fractional hour, 0-24
    cgm = cgm.copy()                                        # don't mutate the original
    cgm["hour_sin"] = np.sin(2 * np.pi * hour / 24)         # cyclical encoding, part 1
    cgm["hour_cos"] = np.cos(2 * np.pi * hour / 24)         # cyclical encoding, part 2

    parts = [cgm]                                                # start table with glucose

    # --- each signal: load, (accelerometer -> magnitude), summarize to grid ---
    for spec in signal_specs:                               # loop the requested signals
        if spec.get("acc"):                                 # accelerometer needs 3-axis handling
            acc = pd.read_csv(folder / f"{spec['name']}_{subject}.csv")  # read ACC
            acc.columns = [c.strip() for c in acc.columns]  # strip column spaces
            acc["datetime"] = pd.to_datetime(acc["datetime"])  # parse time
            acc["motion"] = np.sqrt(acc["acc_x"]**2 + acc["acc_y"]**2 + acc["acc_z"]**2)  # magnitude
            sig = acc[["datetime", "motion"]].dropna().sort_values("datetime").reset_index(drop=True)  # tidy
            col = "motion"                                  # feature name base
        else:                                               # normal 1-column signal
            sig = load_e4_signal(folder, spec["name"], subject, spec["col"],  # reuse loader
                                 clip_range=spec.get("clip"))  # with optional clip
            col = spec["col"]                               # feature name base

        feats = summarize_to_cgm(sig, col, cgm_times,       # reuse summarizer
                                 min_coverage=spec.get("min_cov", min_coverage))  # coverage rule
        feats = feats.drop(columns=f"{col}_coverage")       # drop coverage col (keep features only)
        parts.append(feats)                                 # add this signal's features

    return pd.concat(parts, axis=1)                         # glue glucose + all signals into one table


def summarize_hrv_clean_to_cgm(ibi_df, cgm_times, window_min=5,   # Kubios-corrected HRV per slot
                               min_beats=30, sampling_rate=1000):  # need >=30 beats; ms resolution
    """HRV per 5-min slot using NeuroKit2's Kubios artifact correction.

    Why: E4 detects only ~33% of beats, and misdetections inflate RMSSD/pNN50.
    Kubios (Lipponen & Tarvainen 2019) classifies each beat as ectopic/missed/
    extra/longshort and CORRECTS the peak series before HRV is computed —
    the clinical standard, better than simply excluding bad pairs.

    NOTE: nk.signal_fixpeaks takes PEAKS (sample indices), not RRI. So we
    convert beat timestamps -> peak positions at `sampling_rate` (1000 Hz = ms).

    Returns rmssd/sdnn/pnn50/meannn + diagnostics per CGM timestamp.
    """
    import neurokit2 as nk                                  # imported here (optional dependency)

    times = ibi_df["datetime"].to_numpy()                   # beat timestamps
    window = np.timedelta64(window_min, "m")                # 5-minute slot

    rmssds, sdnns, pnn50s, meannns = [], [], [], []         # HRV features
    counts, n_artifacts = [], []                            # diagnostics

    for t in cgm_times:                                     # each glucose timestamp
        t = np.datetime64(t)                                # numpy datetime
        in_slot = (times > t - window) & (times <= t)       # beats in this slot
        slot_t = times[in_slot]                             # their timestamps

        n = len(slot_t)                                     # beats found
        counts.append(n)                                    # diagnostic

        if n < min_beats:                                   # too few beats -> don't fabricate
            rmssds.append(np.nan); sdnns.append(np.nan)     # all HRV missing
            pnn50s.append(np.nan); meannns.append(np.nan)   #
            n_artifacts.append(np.nan)                      #
            continue                                        # next slot

        # --- convert beat TIMES -> PEAK indices (samples at 1000 Hz = milliseconds) ---
        t0 = slot_t[0]                                      # slot's first beat = time origin
        elapsed_s = (slot_t - t0) / np.timedelta64(1, "s")  # seconds since first beat
        peaks = np.round(elapsed_s * sampling_rate).astype(int)  # -> sample indices (ms)
        peaks = np.unique(peaks)                            # drop any duplicate positions

        if len(peaks) < min_beats:                          # after dedup, still enough?
            rmssds.append(np.nan); sdnns.append(np.nan)     # no -> missing
            pnn50s.append(np.nan); meannns.append(np.nan)   #
            n_artifacts.append(np.nan)                      #
            continue                                        # next slot

        try:                                                # correction can fail on bad input
            # --- Kubios artifact correction (the key step) ---
            info, peaks_clean = nk.signal_fixpeaks(         # returns (artifact info, clean peaks)
                peaks,                                      # peak positions
                sampling_rate=sampling_rate,                # 1000 Hz (ms resolution)
                iterative=True,                             # repeat correction -> better result
                method="Kubios",                            # Lipponen & Tarvainen 2019
                show=False,                                 # no plots
            )

            # count how many beats were flagged as artifacts (diagnostic)
            n_art = sum(len(info.get(k, [])) for k in       # ectopic/missed/extra/longshort
                        ("ectopic", "missed", "extra", "longshort"))

            # --- corrected peaks -> intervals (RRI) in milliseconds ---
            rri = np.diff(peaks_clean) / sampling_rate * 1000  # successive intervals, ms

            # keep only physiologically plausible intervals (40-150 bpm)
            rri = rri[(rri >= 400) & (rri <= 1500)]         # 400ms=150bpm, 1500ms=40bpm

            if len(rri) < min_beats // 2:                   # too few after cleaning
                raise ValueError("too few clean intervals")  # fall through to NaN

            # --- HRV time-domain features on the CORRECTED series ---
            diffs = np.diff(rri)                            # successive differences (ms)
            rmssds.append(np.sqrt(np.mean(diffs ** 2)))     # RMSSD (parasympathetic)
            sdnns.append(rri.std())                         # SDNN (overall variability)
            pnn50s.append(np.mean(np.abs(diffs) > 50) * 100)  # pNN50 (%)
            meannns.append(rri.mean())                      # mean interval (ms)
            n_artifacts.append(n_art)                       # artifacts corrected

        except Exception:                                   # correction failed -> honest NaN
            rmssds.append(np.nan); sdnns.append(np.nan)     # mark missing
            pnn50s.append(np.nan); meannns.append(np.nan)   #
            n_artifacts.append(np.nan)                      #

    return pd.DataFrame({                                   # the feature table
        "hrv_rmssd": rmssds,                                # short-term variability
        "hrv_sdnn": sdnns,                                  # overall variability
        "hrv_pnn50": pnn50s,                                # % successive diffs > 50ms
        "hrv_meannn": meannns,                              # mean interval
        "hrv_n_beats": counts,                              # beats found (diagnostic)
        "hrv_n_artifacts": n_artifacts,                     # beats corrected (diagnostic)
    })