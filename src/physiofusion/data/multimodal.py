"""Build the multimodal windowed dataset across all subjects and save it.
Procedural for now (refactor to config/OO later when the ablation phase needs it).
"""
import numpy as np                                          # arrays
import pandas as pd                                         # data
from physiofusion.features import build_subject_table       # per-subject feature table
from physiofusion.windowing import segment_by_gaps, make_multimodal_windows  # segment + window


def build_multimodal_dataset(folder_root, subjects, signal_specs,  # main builder
                             history=24, horizon=6, step=1, gap_threshold_min=15):
    """Build (X, y, groups, feature_names) across all subjects.

    Steps per subject: aligned table -> drop rows with ANY missing feature ->
    re-segment into gap-free islands (dropping rows creates new gaps!) ->
    window each island -> tag with subject. Then stack everyone.
    """
    # figure out the wristband feature column names from the first subject's table
    sample = build_subject_table(folder_root / subjects[0], subjects[0], signal_specs)  # one table
    feature_cols = [c for c in sample.columns if c not in ("datetime", "glucose")]  # all feature cols

    X_parts, y_parts, g_parts = [], [], []                  # accumulators across subjects

    for subject in subjects:                                # loop every subject
        table = build_subject_table(folder_root / subject, subject, signal_specs)  # aligned table

        before = len(table)                                 # rows before dropping
        table = table.dropna().reset_index(drop=True)       # DROP any slot missing a feature (honest)
        after = len(table)                                  # rows after dropping

        # dropping rows breaks continuity -> re-segment into gap-free islands.
        # We reuse segment_by_gaps on the (now-thinned) timestamps.
        table = segment_by_gaps(table.rename(columns={"datetime": "ts"}),  # segmenter expects 'ts'
                                gap_threshold_min=gap_threshold_min)  # re-cut into islands

        # window each island separately (never across a gap)
        subj_windows = 0                                    # count for reporting
        for _, island in table.groupby("island"):           # each continuous island
            Xi, yi = make_multimodal_windows(island, feature_cols, history, horizon, step)  # window it
            if len(Xi):                                     # if it produced windows
                X_parts.append(Xi)                          # add histories+features
                y_parts.append(yi)                          # add targets
                g_parts.append(np.full(len(Xi), subject, dtype=object))  # tag subject
                subj_windows += len(Xi)                     # tally

        print(f"  {subject}: {before}->{after} slots kept, {subj_windows} windows")  # progress

    X = np.concatenate(X_parts)                             # stack all histories+features
    y = np.concatenate(y_parts)                             # stack all targets
    groups = np.concatenate(g_parts)                        # stack all subject tags

    # feature_names = the 24 glucose lags + the wristband feature columns
    glucose_names = [f"glucose_lag_{i}" for i in range(history)]  # name the history columns
    feature_names = glucose_names + feature_cols            # full column naming

    return X, y, groups, feature_names                      # everything the models need


def save_dataset(X, y, groups, feature_names, out_dir, name="multimodal"):  # save to Parquet
    """Save the windowed dataset as Parquet (+ a small metadata note)."""
    out_dir.mkdir(parents=True, exist_ok=True)             # ensure output folder exists
    # build a DataFrame: feature columns + target + group, one row per window
    df = pd.DataFrame(X, columns=feature_names)             # features
    df["y"] = y                                            # target column
    df["subject"] = groups                                 # subject tag column
    path = out_dir / f"{name}.parquet"                     # output path
    df.to_parquet(path, index=False)                       # write Parquet (typed, compressed)
    print(f"saved {len(df)} windows x {len(feature_names)} features -> {path}")  # confirm
    return path        
                                    # return where we saved


def build_sequence_dataset(folder_root, subjects, signal_specs, channel_cols,  # sequential builder
                           history=24, horizon=6, step=1, gap_threshold_min=15):
    """Build (X, y, groups) as multi-channel SEQUENCES for early fusion.

    Same honest pipeline as before: aligned table -> drop rows with ANY missing
    feature -> re-segment into gap-free islands -> window each island.
    Difference: windows keep the full history of every channel.

    channel_cols : which columns become channels, e.g.
                   ["glucose","hr_mean","eda_mean","temp_mean","motion_mean"]
    Returns X (n, n_channels, history), y (n,), groups (n,)
    """
    from physiofusion.windowing import make_sequence_windows   # the new windower

    X_parts, y_parts, g_parts = [], [], []                     # accumulators across subjects

    for subject in subjects:                                   # loop every subject
        table = build_subject_table(folder_root / subject, subject, signal_specs)  # aligned table

        before = len(table)                                    # rows before dropping
        table = table.dropna().reset_index(drop=True)          # drop slots missing any signal
        after = len(table)                                     # rows after dropping

        # dropping rows breaks continuity -> re-segment into gap-free islands
        table = segment_by_gaps(table.rename(columns={"datetime": "ts"}),  # segmenter needs 'ts'
                                gap_threshold_min=gap_threshold_min)       # re-cut islands

        subj_windows = 0                                       # count for reporting
        for _, island in table.groupby("island"):              # window each island separately
            Xi, yi = make_sequence_windows(island, channel_cols, history, horizon, step)  # sequences
            if len(Xi):                                        # if this island produced windows
                X_parts.append(Xi)                             # add (n, ch, history) block
                y_parts.append(yi)                             # add targets
                g_parts.append(np.full(len(Xi), subject, dtype=object))  # tag subject
                subj_windows += len(Xi)                        # tally

        print(f"  {subject}: {before}->{after} slots kept, {subj_windows} windows")  # progress

    X = np.concatenate(X_parts)                                # (N, n_channels, history)
    y = np.concatenate(y_parts)                                # (N,)
    groups = np.concatenate(g_parts)                           # (N,)
    return X, y, groups                                        # ready for the TCN


def save_sequence_dataset(X, y, groups, out_dir, name="multimodal_seq"):  # save 3D arrays
    """Save sequence dataset as .npy files (Parquet is for flat tables, not 3D)."""
    out_dir.mkdir(parents=True, exist_ok=True)                 # ensure folder exists
    np.save(out_dir / f"{name}_X.npy", X)                      # features (N, ch, history)
    np.save(out_dir / f"{name}_y.npy", y)                      # targets (N,)
    np.save(out_dir / f"{name}_groups.npy", groups)            # subject tags (N,)
    print(f"saved {X.shape} sequences -> {out_dir}/{name}_*.npy")  # confirm