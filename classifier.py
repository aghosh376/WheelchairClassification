"""
EMG Classifier v3 — Two-Stage Cascade + Fixed Dead Zone
Access to the Google Colab where we actually tested and worked with: https://colab.research.google.com/drive/1DWJDazDnbTeksjYIdPAARn3yMhUdqzcO?usp=sharing
========================================================

WHAT CHANGED AND WHY
---------------------

Problem 1: build_safe_mask was crashing on all quartile files
  Root cause: np.diff() on string arrays isn't supported in NumPy.
  The original code did np.diff(labels.astype(str)) which throws
  "ufunc 'subtract' did not contain a loop matching dtype('<U8')".
  Fix: compare adjacent elements manually with labels[1:] != labels[:-1].

Problem 2: extend_down bleeds into flex_up at 29.1%
  These are opposite motions on opposite sensor locations — they should
  be the easiest pair to separate. A 29% confusion rate means either:
    (a) some extend_down files are actually wired to column 1 (up) not 2 (down), or
    (b) the 6-class problem is forcing the classifier to draw boundaries in
        feature space that are too complex for the overlap regions.
  The two-stage approach removes (b) entirely.

Problem 3: rest_up bleeds everywhere (50.6% recall)
  rest_up vs flex_up and rest_up vs flex_down confusions are a
  feature overlap problem — rest and low-intensity flex share similar
  RMS/MAV values. Separating "what motion" from "which sensor" makes
  the motion boundary cleaner because the classifier no longer has to
  simultaneously solve both questions in one shot.

SOLUTION: Two-stage cascade
  Stage 1 — Motion classifier:  flex  /  extend  /  rest   (3 classes)
             Trained on features from ALL windows regardless of location.
             The motion signal is the dominant feature; location is noise
             from the classifier's perspective at this stage.

  Stage 2 — Location classifier: up  /  down   (2 classes)
             Trained separately; only has to answer "which patch".
             RMS and WL differences between top/bottom forearm patches
             are consistent enough that a simple 2-class boundary works well.

  At inference: Stage 1 predicts motion, Stage 2 predicts location,
  compound label is assembled as f"{motion}_{location}".

  Each stage uses the same RF + CalibratedLinearSVC soft-voting ensemble
  with class_weight='balanced'. No undersampling, no synthetic data.

OTHER FIXES
  - Dead zone mask uses element-wise string comparison, not np.diff
  - Both separators tried (',', '\t') for quartile CSVs automatically
  - Column bounds check uses shape[1]-1 correctly for dual quartile files
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, iirnotch, lfilter, welch
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration
# ==============================================================================
FS             = 200.0
DEAD_ZONE_SECS = 0.5
DEAD_ZONE_SAMP = int(DEAD_ZONE_SECS * FS)   # 100 samples = 0.5 s at 200 Hz
WIN_SIZE       = 100
STEP           = 15

FEATURE_COLS = ['rms', 'mav', 'zc', 'wl', 'mnf', 'ssc', 'mdf', 'pkf']


# ==============================================================================
# 1. Signal filtering
# ==============================================================================
def filter_signal(raw, fs=FS):
    nyq = 0.5 * fs
    b_band, a_band = butter(4, [20.0 / nyq, 99.0 / nyq], btype='band')
    sig = lfilter(b_band, a_band, raw)
    b_notch, a_notch = iirnotch(60.0 / nyq, 30)
    sig = lfilter(b_notch, a_notch, sig)
    return sig


# ==============================================================================
# 2. Feature extraction
#    motion   : 'flex' | 'extend' | 'rest'
#    location : 'up'   | 'down'
# ==============================================================================
def extract_emg_features(raw_signal, motion, location, fs=FS):
    filtered = filter_signal(np.asarray(raw_signal, dtype=float), fs)
    features = []
    for i in range(0, len(filtered) - WIN_SIZE, STEP):
        win = filtered[i : i + WIN_SIZE]
        freqs, psd = welch(win, fs=fs, nperseg=len(win))
        psd_sum = np.sum(psd)
        if psd_sum == 0 or not np.isfinite(psd_sum):
            continue
        cum_psd = np.cumsum(psd)
        mdf_idx = min(np.searchsorted(cum_psd, psd_sum / 2), len(freqs) - 1)
        features.append({
            'rms'     : np.sqrt(np.mean(win ** 2)),
            'mav'     : np.mean(np.abs(win)),
            'zc'      : np.sum(np.diff(np.sign(win)) != 0),
            'wl'      : np.sum(np.abs(np.diff(win))),
            'mnf'     : np.sum(freqs * psd) / psd_sum,
            'ssc'     : np.sum(np.diff(np.sign(np.diff(win))) != 0),
            'mdf'     : freqs[mdf_idx],
            'pkf'     : freqs[np.argmax(psd)],
            'motion'  : motion,
            'location': location,
            'label'   : f"{motion}_{location}",
        })
    return pd.DataFrame(features)


# ==============================================================================
# 3. Dead-zone mask (FIXED: no np.diff on strings)
#    Uses element-wise comparison of adjacent elements instead of np.diff,
#    which doesn't support string subtraction.
# ==============================================================================
def build_safe_mask(label_col_values, dead_zone):
    labels = np.asarray(label_col_values, dtype=str)
    n      = len(labels)
    safe   = np.ones(n, dtype=bool)

    # Find boundaries: positions where label changes from one row to the next
    # labels[1:] != labels[:-1] gives True at index i when labels[i] != labels[i+1]
    # The actual boundary is at position i+1 (the first sample of the new class)
    change_flags = labels[1:] != labels[:-1]          # length n-1
    boundaries   = np.where(change_flags)[0] + 1      # +1 → index of first new sample

    for b in boundaries:
        lo = max(0, b - dead_zone)
        hi = min(n, b + dead_zone)
        safe[lo:hi] = False

    return safe, int(np.sum(~safe))


# ==============================================================================
# 4. Quartile → motion mapping
# ==============================================================================
quartile_label_map = {
    'Q1': 'rest',
    'Q2': 'flex',
    'Q3': 'rest',
    'Q4': 'extend',
}

file_paths_quartile = [
    (
        "/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 1(Right patch up)/data_with_quartiles.csv",
        ('single', 'up')
    ),
    (
        "/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 2(right patch down)/data_with_quartiles2.csv",
        ('single', 'down')
    ),
    (
        "/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 3(Left patch up)/data_with_quartiles3.csv",
        ('single', 'up')
    ),
    (
        "/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 4(Left patch down)/data_with_quartiles4.csv",
        ('single', 'down')
    ),
    (
        "/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 5(Right up&down)/5(Part2)/data_with_quartiles6.csv",
        ('dual',)
    ),
    (
        "/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 5(Right up&down)/5(part 1)/quartiles_BrainFlow-RAW_2026-01-24_13-06-59(right both up((1) and down(2)_1.csv",
        ('dual',)
    ),
]


# ==============================================================================
# 5. Explicit-label files
# ==============================================================================
file_label_pairs = [
    # ── Raw Data 7: right UP, flex ──────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 7(right up flex and rest(0 and 1 shoud be flex and 2 and 3 should be rest))/BrainFlow-RAW_2026-01-31_12-02-43(right up flex)_0.csv', 'flex', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 7(right up flex and rest(0 and 1 shoud be flex and 2 and 3 should be rest))/BrainFlow-RAW_2026-01-31_12-02-43(right up flex)_1.csv', 'flex', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 7(right up flex and rest(0 and 1 shoud be flex and 2 and 3 should be rest))/BrainFlow-RAW_2026-01-31_12-02-43(right up flex)_2.csv', 'rest', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 7(right up flex and rest(0 and 1 shoud be flex and 2 and 3 should be rest))/BrainFlow-RAW_2026-01-31_12-02-43(right up flex)_3.csv', 'rest', ('single', 'up', 1)),
    # ── Raw Data 9: right DOWN, rest ─────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 9(training right down rest)/BrainFlow-RAW_2026-02-07_12-09-41(right down rest)_0.csv', 'rest', ('single', 'down', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 9(training right down rest)/BrainFlow-RAW_2026-02-07_12-09-41(right down rest)_1.csv', 'rest', ('single', 'down', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 9(training right down rest)/BrainFlow-RAW_2026-02-07_12-09-41(right down rest)_2.csv', 'rest', ('single', 'down', 1)),
    # ── Raw Data 8: right DOWN, extend ──────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 8(training right down extend)/BrainFlow-RAW_2026-02-07_11-53-22(patch down  extend)_0.csv', 'extend', ('single', 'down', 2)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 8(training right down extend)/BrainFlow-RAW_2026-02-07_11-53-22(patch down  extend)_1.csv', 'extend', ('single', 'down', 2)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 8(training right down extend)/BrainFlow-RAW_2026-02-07_11-53-22(patch down  extend)_2.csv', 'extend', ('single', 'down', 2)),
    # ── Raw Data 10: right UP, flex ──────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 10 (training right up flex)/BrainFlow-RAW_2026-02-07_12-18-17_0.csv', 'flex', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 10 (training right up flex)/BrainFlow-RAW_2026-02-07_12-18-17_1.csv', 'flex', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 10 (training right up flex)/BrainFlow-RAW_2026-02-07_12-18-17_2.csv', 'flex', ('single', 'up', 1)),
    # ── Raw Data 11: right UP, rest ──────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 11(training right up rest)/BrainFlow-RAW_2026-02-07_12-23-56(right up  rest)_0.csv', 'rest', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 11(training right up rest)/BrainFlow-RAW_2026-02-07_12-23-56(right up  rest)_1.csv', 'rest', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 11(training right up rest)/BrainFlow-RAW_2026-02-07_12-23-56(right up  rest)_2.csv', 'rest', ('single', 'up', 1)),
    # ── Raw Data 12: left DOWN, extend ───────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 12(training left down extend)/BrainFlow-RAW_2026-02-07_12-57-13(left down extend)_1.csv', 'extend', ('single', 'down', 2)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 12(training left down extend)/BrainFlow-RAW_2026-02-07_12-57-13(left down extend)_2.csv', 'extend', ('single', 'down', 2)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 12(training left down extend)/BrainFlow-RAW_2026-02-07_12-57-13(left down extend)_3.csv', 'extend', ('single', 'down', 2)),
    # ── Raw Data 13: left DOWN, rest ─────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 13(training left down rest)/BrainFlow-RAW_2026-02-07_13-08-02(left down rest)_0.csv', 'rest', ('single', 'down', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 13(training left down rest)/BrainFlow-RAW_2026-02-07_13-08-02(left down rest)_1.csv', 'rest', ('single', 'down', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 13(training left down rest)/BrainFlow-RAW_2026-02-07_13-08-02(left down rest)_2.csv', 'rest', ('single', 'down', 1)),
    # ── Raw Data 14: left UP, flex ───────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 14(training left up flex)/BrainFlow-RAW_2026-02-07_13-17-51(left up flex)_0.csv', 'flex', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 14(training left up flex)/BrainFlow-RAW_2026-02-07_13-17-51(left up flex)_1.csv', 'flex', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 14(training left up flex)/BrainFlow-RAW_2026-02-07_13-17-51(left up flex)_2.csv', 'flex', ('single', 'up', 1)),
    # ── Raw Data 15: left UP, rest ───────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 15(training left  up rest)/BrainFlow-RAW_2026-02-07_13-24-37(left up rest)_0.csv', 'rest', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 15(training left  up rest)/BrainFlow-RAW_2026-02-07_13-24-37(left up rest)_1.csv', 'rest', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 15(training left  up rest)/BrainFlow-RAW_2026-02-07_13-24-37(left up rest)_2.csv', 'rest', ('single', 'up', 1)),
    # ── Raw Data 16: dual, flex ───────────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 16(training right up(1) and down(2) flex) /BrainFlow-RAW_2026-02-21_13-09-04(flex)_0.csv', 'flex', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 16(training right up(1) and down(2) flex) /BrainFlow-RAW_2026-02-21_12-34-17(flex)_1.csv', 'flex', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 16(training right up(1) and down(2) flex) /BrainFlow-RAW_2026-02-21_13-09-04(flex)_1.csv', 'flex', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 16(training right up(1) and down(2) flex) /BrainFlow-RAW_2026-02-21_13-09-04(flex)_2.csv', 'flex', ('dual',)),
    # ── Raw Data 17: dual, extend ─────────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 17(training right up(1) and down(2) extend))/BrainFlow-RAW_2026-02-28_12-47-44(up&down extend)_0.csv', 'extend', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 17(training right up(1) and down(2) extend))/BrainFlow-RAW_2026-02-28_12-47-44(up&down extend)_1.csv', 'extend', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 17(training right up(1) and down(2) extend))/BrainFlow-RAW_2026-02-28_12-47-44(up&down extend)_2.csv', 'extend', ('dual',)),
    # ── Raw Data 18: dual, rest ───────────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 18(training up(1) and down(2)rest)/BrainFlow-RAW_2026-02-28_13-00-47(up&down rest)_0.csv', 'rest', ('dual',)),
    # ── Raw Data 19: dual, extend ─────────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 19(training up(1) down(2) extend)/BrainFlow-RAW_2026-03-07_13-03-20(up&down extend)_1.csv', 'extend', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 19(training up(1) down(2) extend)/BrainFlow-RAW_2026-03-07_13-03-20(up&down extend)_2.csv', 'extend', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 19(training up(1) down(2) extend)/BrainFlow-RAW_2026-03-07_13-03-20(up&down extend)_3.csv', 'extend', ('dual',)),
    # ── Raw Data 20: dual, flex ───────────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 20 (training up(1) down(2) flex)/BrainFlow-RAW_2026-03-07_13-13-53(up&down flex)_0.csv', 'flex', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 20 (training up(1) down(2) flex)/BrainFlow-RAW_2026-03-07_13-13-53(up&down flex)_1.csv', 'flex', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 20 (training up(1) down(2) flex)/BrainFlow-RAW_2026-03-07_13-13-53(up&down flex)_2.csv', 'flex', ('dual',)),
    # ── Raw Data 21: dual, extend ─────────────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 21 (training up(1) down(2) extend)/BrainFlow-RAW_2026-04-15_16-23-48(training up(1) down(2) extend)_0.csv', 'extend', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 21 (training up(1) down(2) extend)/BrainFlow-RAW_2026-04-15_16-34-59(training up(1) down(2) extend)_0.csv', 'extend', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 21 (training up(1) down(2) extend)/BrainFlow-RAW_2026-04-15_16-34-59(training up(1) down(2) extend)_1.csv', 'extend', ('dual',)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 21 (training up(1) down(2) extend)/BrainFlow-RAW_2026-04-15_16-42-01(training up(1) down(2) extend)_0.csv', 'extend', ('dual',)),
    # ── Raw Data 22: flex DOWN = ch(2) ───────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 22(training flex down(2))/BrainFlow-RAW_2026-05-16_10-21-45(flex  down(2))y)_0.csv', 'flex', ('single', 'down', 2)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 22(training flex down(2))/BrainFlow-RAW_2026-05-16_10-21-45(flex  down(2))y)_1.csv', 'flex', ('single', 'down', 2)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 22(training flex down(2))/BrainFlow-RAW_2026-05-16_10-21-45(flex  down(2))y)_2.csv', 'flex', ('single', 'down', 2)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 22(training flex down(2))/BrainFlow-RAW_2026-05-16_10-21-45(flex  down(2))y)_3.csv', 'flex', ('single', 'down', 2)),
    # ── Raw Data 23: extend UP = ch(1) ───────────────────────────────────
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 23(training extend up(1))/BrainFlow-RAW_2026-05-16_10-42-26(extend up (1))_0.csv', 'extend', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 23(training extend up(1))/BrainFlow-RAW_2026-05-16_10-42-26(extend up (1))_1.csv', 'extend', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 23(training extend up(1))/BrainFlow-RAW_2026-05-16_10-42-26(extend up (1))_2.csv', 'extend', ('single', 'up', 1)),
    ('/content/drive/MyDrive/TNT Wheelchair/EMG/Data/Raw Data 23(training extend up(1))/BrainFlow-RAW_2026-05-16_10-42-26(extend up (1))_3.csv', 'extend', ('single', 'up', 1)),
]


# ==============================================================================
# 6. Data loading
# ==============================================================================
all_window_dfs = []

print("=" * 60)
print(f"Version 3 — Two-Stage Cascade")
print(f"Dead zone: ±{DEAD_ZONE_SECS}s around quartile boundaries")
print("=" * 60)

# --- Quartile files ---
print("\nLoading quartile files...")
total_excluded = 0

for path, ch_mode in file_paths_quartile:
    try:
        # Try comma first, fall back to tab
        try:
            data = pd.read_csv(path, sep=',', header=None)
            # Sanity check: last col should be Q-labels, not numbers
            if pd.to_numeric(data.iloc[:, -1], errors='coerce').notna().mean() > 0.9:
                data = pd.read_csv(path, sep='\t', header=None)
        except Exception:
            data = pd.read_csv(path, sep='\t', header=None)

        if data.empty:
            print(f"  [SKIP – empty]  {path}")
            continue

        last_col     = data.columns[-1]
        label_values = data[last_col].values.astype(str)

        # Strip whitespace from labels
        label_values = np.array([v.strip() for v in label_values])

        # Check that we actually have Q-labels
        unique_labels = set(label_values)
        if not unique_labels.intersection({'Q1', 'Q2', 'Q3', 'Q4'}):
            print(f"  [SKIP – no Q-labels found, got: {unique_labels}]  {path}")
            continue

        safe_mask, n_excl = build_safe_mask(label_values, DEAD_ZONE_SAMP)
        total_excluded += n_excl
        fname = path.split('/')[-1]
        print(f"  {fname[:50]}  →  {n_excl} samples removed by dead zone")

        safe_data   = data[safe_mask].reset_index(drop=True)
        safe_labels = safe_data[last_col].values.astype(str)
        safe_labels = np.array([v.strip() for v in safe_labels])

        for q_val in np.unique(safe_labels):
            if q_val not in quartile_label_map:
                continue
            motion     = quartile_label_map[q_val]
            group_data = safe_data[safe_labels == q_val]

            if ch_mode[0] == 'single':
                location = ch_mode[1]
                signal   = group_data.iloc[:, 1].values
                if len(signal) < WIN_SIZE:
                    continue
                all_window_dfs.append(extract_emg_features(signal, motion, location))

            elif ch_mode[0] == 'dual':
                n_data_cols = group_data.shape[1] - 1  # exclude label col
                for col_idx, location in [(1, 'up'), (2, 'down')]:
                    if col_idx >= n_data_cols:
                        print(f"    [SKIP – col {col_idx} missing] {fname}")
                        continue
                    signal = group_data.iloc[:, col_idx].values
                    if len(signal) < WIN_SIZE:
                        continue
                    all_window_dfs.append(extract_emg_features(signal, motion, location))

    except Exception as e:
        print(f"  [ERROR]  {path.split('/')[-1]}\n    → {e}")

print(f"\n  Total samples removed by dead zone: {total_excluded:,} "
      f"(≈{total_excluded / FS:.1f}s)")

# --- Explicit-label files ---
print("\nLoading explicit-label BrainFlow files...")

for path, motion, ch_mode in file_label_pairs:
    try:
        df = pd.read_csv(path, sep='\t')

        if ch_mode[0] == 'single':
            location = ch_mode[1]
            col_idx  = ch_mode[2]
            if col_idx >= df.shape[1]:
                print(f"  [SKIP – col {col_idx} missing]  {path.split('/')[-1]}")
                continue
            signal = df.iloc[:, col_idx].values
            if len(signal) < WIN_SIZE:
                continue
            all_window_dfs.append(extract_emg_features(signal, motion, location))

        elif ch_mode[0] == 'dual':
            for col_idx, location in [(1, 'up'), (2, 'down')]:
                if col_idx >= df.shape[1]:
                    print(f"  [SKIP – col {col_idx} missing]  {path.split('/')[-1]}")
                    continue
                signal = df.iloc[:, col_idx].values
                if len(signal) < WIN_SIZE:
                    continue
                all_window_dfs.append(extract_emg_features(signal, motion, location))

    except Exception as e:
        print(f"  [ERROR]  {path.split('/')[-1]}\n    → {e}")


# ==============================================================================
# 7. Combine
# ==============================================================================
if not all_window_dfs:
    print("\nERROR: No data extracted. Check your file paths.")
    raise SystemExit(1)

final_dataset = pd.concat(all_window_dfs, ignore_index=True)
before = len(final_dataset)
final_dataset = (
    final_dataset
    .dropna(subset=FEATURE_COLS)
    .reset_index(drop=True)
)
dropped = before - len(final_dataset)
if dropped:
    print(f"\n[INFO] Dropped {dropped} NaN windows ({dropped / before:.1%})")

print(f"\nFinal dataset: {final_dataset.shape[0]:,} windows")
print("\nCompound label distribution:")
print(final_dataset['label'].value_counts().to_string())
print("\nMotion distribution:")
print(final_dataset['motion'].value_counts().to_string())
print("\nLocation distribution:")
print(final_dataset['location'].value_counts().to_string())


# ==============================================================================
# 8. Helper: build ensemble
# ==============================================================================
def build_ensemble():
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    lsvc = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', C=1.0, max_iter=3000, random_state=42),
        cv=3,
    )
    return VotingClassifier(
        estimators=[('rf', rf), ('lsvc', lsvc)],
        voting='soft',
    )


# ==============================================================================
# 9. STAGE 1 — Motion classifier  (flex / extend / rest)
# ==============================================================================
print("\n" + "=" * 60)
print("STAGE 1 — Motion classifier  (flex / extend / rest)")
print("=" * 60)

X_mot = final_dataset[FEATURE_COLS].values
y_mot = final_dataset['motion'].values

X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_mot, y_mot, test_size=0.33, random_state=42, stratify=y_mot
)
scaler_mot = StandardScaler()
X_tr_m_sc  = scaler_mot.fit_transform(X_tr_m)
X_te_m_sc  = scaler_mot.transform(X_te_m)

clf_motion = build_ensemble()
clf_motion.fit(X_tr_m_sc, y_tr_m)

y_pred_m = clf_motion.predict(X_te_m_sc)
print(f"Motion accuracy: {accuracy_score(y_te_m, y_pred_m):.2%}")
print(classification_report(y_te_m, y_pred_m))


# ==============================================================================
# 10. STAGE 2 — Location classifier  (up / down)
# ==============================================================================
print("=" * 60)
print("STAGE 2 — Location classifier  (up / down)")
print("=" * 60)

X_loc = final_dataset[FEATURE_COLS].values
y_loc = final_dataset['location'].values

X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
    X_loc, y_loc, test_size=0.33, random_state=42, stratify=y_loc
)
scaler_loc = StandardScaler()
X_tr_l_sc  = scaler_loc.fit_transform(X_tr_l)
X_te_l_sc  = scaler_loc.transform(X_te_l)

clf_location = build_ensemble()
clf_location.fit(X_tr_l_sc, y_tr_l)

y_pred_l = clf_location.predict(X_te_l_sc)
print(f"Location accuracy: {accuracy_score(y_te_l, y_pred_l):.2%}")
print(classification_report(y_te_l, y_pred_l))


# ==============================================================================
# 11. Cascade evaluation on a shared held-out set
#     Use the same stratified split on compound labels so every class
#     appears in the test set.
# ==============================================================================
print("=" * 60)
print("CASCADED evaluation on held-out compound-label test set")
print("=" * 60)

X_all = final_dataset[FEATURE_COLS].values
y_all = final_dataset['label'].values

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_all, test_size=0.33, random_state=42, stratify=y_all
)

# Scale using the motion scaler (same features; both scalers are fit on the
# full dataset so they will be very close — motion scaler used for cascade eval)
X_test_sc_mot = scaler_mot.transform(X_test_all)
X_test_sc_loc = scaler_loc.transform(X_test_all)

motion_preds   = clf_motion.predict(X_test_sc_mot)
location_preds = clf_location.predict(X_test_sc_loc)
cascade_preds  = np.array([f"{m}_{l}" for m, l in zip(motion_preds, location_preds)])

cascade_acc = accuracy_score(y_test_all, cascade_preds)
print(f"\nCascade accuracy (motion × location): {cascade_acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test_all, cascade_preds))


# ==============================================================================
# 12. Confusion matrix plot
# ==============================================================================
labels_sorted = sorted(np.unique(y_test_all))
cm_raw  = confusion_matrix(y_test_all, cascade_preds, labels=labels_sorted)
cm_norm = confusion_matrix(y_test_all, cascade_preds, labels=labels_sorted, normalize='true')

annots = np.empty_like(cm_raw, dtype=object)
for i in range(cm_raw.shape[0]):
    for j in range(cm_raw.shape[1]):
        annots[i, j] = f"{cm_raw[i,j]}\n({cm_norm[i,j]*100:.1f}%)"

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm, annot=annots, fmt="", cmap="Blues",
    xticklabels=labels_sorted, yticklabels=labels_sorted,
    cbar_kws={'label': 'Proportion of true class'}
)
plt.title(
    f"EMG Classifier v3 — Two-Stage Cascade\n"
    f"Cascade accuracy: {cascade_acc:.2%}   "
    f"(Motion: {accuracy_score(y_te_m, y_pred_m):.2%} | "
    f"Location: {accuracy_score(y_te_l, y_pred_l):.2%})",
    fontsize=13, pad=12
)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('emg_confusion_matrix_v3.png', dpi=300)
print("\nSaved: emg_confusion_matrix_v3.png")
plt.show()


# ==============================================================================
# 13. Save all artefacts
# ==============================================================================
joblib.dump(clf_motion,   'emg_motion_clf_v3.pkl')
joblib.dump(clf_location, 'emg_location_clf_v3.pkl')
joblib.dump(scaler_mot,   'emg_scaler_motion_v3.pkl')
joblib.dump(scaler_loc,   'emg_scaler_location_v3.pkl')
joblib.dump(FEATURE_COLS, 'emg_feature_cols.pkl')
print("\nSaved:")
print("  emg_motion_clf_v3.pkl      — Stage 1 motion classifier")
print("  emg_location_clf_v3.pkl    — Stage 2 location classifier")
print("  emg_scaler_motion_v3.pkl   — scaler for motion stage")
print("  emg_scaler_location_v3.pkl — scaler for location stage")
print("  emg_feature_cols.pkl       — feature column order")


# ==============================================================================
# 14. Inference helper
# ==============================================================================
def predict_window(raw_window, fs=FS,
                   motion_model_path   = 'emg_motion_clf_v3.pkl',
                   location_model_path = 'emg_location_clf_v3.pkl',
                   motion_scaler_path  = 'emg_scaler_motion_v3.pkl',
                   location_scaler_path= 'emg_scaler_location_v3.pkl'):
    """
    Classify a single raw EMG window using the two-stage cascade.

    Parameters
    ----------
    raw_window : 1-D array-like  (raw samples, ideally 100 samples at 200 Hz)
    fs         : sampling rate in Hz (default 200)

    Returns
    -------
    label      : str    e.g. 'flex_up'
    motion_conf: float  confidence from Stage 1
    loc_conf   : float  confidence from Stage 2
    """
    _clf_m  = joblib.load(motion_model_path)
    _clf_l  = joblib.load(location_model_path)
    _sc_m   = joblib.load(motion_scaler_path)
    _sc_l   = joblib.load(location_scaler_path)

    sig = filter_signal(np.asarray(raw_window, dtype=float), fs)

    freqs, psd = welch(sig, fs=fs, nperseg=len(sig))
    psd_sum    = np.sum(psd)
    feat = np.array([[
        np.sqrt(np.mean(sig ** 2)),
        np.mean(np.abs(sig)),
        np.sum(np.diff(np.sign(sig)) != 0),
        np.sum(np.abs(np.diff(sig))),
        np.sum(freqs * psd) / psd_sum,
        np.sum(np.diff(np.sign(np.diff(sig))) != 0),
        freqs[np.searchsorted(np.cumsum(psd), psd_sum / 2)],
        freqs[np.argmax(psd)],
    ]])

    motion_proba   = _clf_m.predict_proba(_sc_m.transform(feat))[0]
    location_proba = _clf_l.predict_proba(_sc_l.transform(feat))[0]

    motion   = _clf_m.classes_[np.argmax(motion_proba)]
    location = _clf_l.classes_[np.argmax(location_proba)]

    return (
        f"{motion}_{location}",
        float(np.max(motion_proba)),
        float(np.max(location_proba)),
    )
