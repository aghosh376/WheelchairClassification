"""
live_predict.py — Single-channel EMG prediction (debug / fallback).
Use this to verify one electrode is working before running multi_channel_predict.py.

Channel used: sample[0] (first Ganglion pin).
"""
import time
import numpy as np
import os
import joblib
from collections import deque
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import butter, iirnotch, lfilter, welch

script_dir = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────
# 1. Load model
# ──────────────────────────────────────────────────────────────
model_path  = os.path.join(script_dir, 'emg_svm_model.pkl')
scaler_path = os.path.join(script_dir, 'emg_scaler.pkl')

try:
    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model loaded. Classes: {clf.classes_}")
except FileNotFoundError:
    print(f"[ERROR] .pkl files not found in: {script_dir}")
    exit()

class_labels = list(clf.classes_)
idx_flex = class_labels.index('flex')
idx_ext  = class_labels.index('extend')
idx_rest = class_labels.index('rest')

# ──────────────────────────────────────────────────────────────
# 2. Signal processing — must match training exactly
# ──────────────────────────────────────────────────────────────
fs   = 200.0
nyq  = 0.5 * fs
b_band,  a_band  = butter(4, [20.0/nyq, 99.0/nyq], btype='band')
b_notch, a_notch = iirnotch(60.0/nyq, 30)

WINDOW_SIZE        = 100    # ← must match training (was wrongly 50 before)
FILTER_BUFFER_SIZE = 250
SCALE_FACTOR       = 1_000_000.0   # V → µV, match training units
CONFIDENCE_THRESH  = 0.75

emg_buffer = deque(maxlen=FILTER_BUFFER_SIZE)

# ──────────────────────────────────────────────────────────────
# 3. Feature extraction — identical to training
# ──────────────────────────────────────────────────────────────
def extract_features(win, fs=200.0):
    """Extract 6 features from a 1D window. Order must match training: [rms, mav, zc, wl, mnf, ssc]"""
    rms = np.sqrt(np.mean(win**2))
    mav = np.mean(np.abs(win))
    zc  = np.sum(np.diff(np.sign(win)) != 0)
    wl  = np.sum(np.abs(np.diff(win)))
    ssc = np.sum(np.diff(np.sign(np.diff(win))) != 0)
    freqs, psd = welch(win, fs=fs, nperseg=len(win))
    psd_sum = np.sum(psd) or 1e-10
    mnf = np.sum(freqs * psd) / psd_sum
    return np.array([[rms, mav, zc, wl, mnf, ssc]])  # shape: (1, 6)

# ──────────────────────────────────────────────────────────────
# 4. Connect to LSL
# ──────────────────────────────────────────────────────────────
def connect(target_name='obci_eeg1', target_type='EEG', timeout=5.0):
    print("Scanning for stream...")
    streams = resolve_byprop('name', target_name, timeout=timeout)
    if not streams:
        streams = resolve_byprop('type', target_type, timeout=timeout)
    if not streams:
        print("[ERROR] No stream found.")
        exit()
    inlet = StreamInlet(streams[0])
    print(f"[OK] Connected: {streams[0].name()}")
    return inlet

inlet = connect()

# ──────────────────────────────────────────────────────────────
# 5. Prediction loop
# ──────────────────────────────────────────────────────────────
print(f"\nRunning single-channel prediction on Ch0 (window={WINDOW_SIZE}, thresh={CONFIDENCE_THRESH:.0%})")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        chunk, _ = inlet.pull_chunk()

        if chunk:
            for sample in chunk:
                emg_buffer.append(sample[0] * SCALE_FACTOR)  # Ch0 only, scaled to µV

            if len(emg_buffer) == FILTER_BUFFER_SIZE:
                raw = np.array(emg_buffer)
                raw -= np.mean(raw)  # remove DC offset

                filtered = lfilter(b_band,  a_band,  raw)
                filtered = lfilter(b_notch, a_notch, filtered)

                win = filtered[-WINDOW_SIZE:]

                features_scaled = scaler.transform(extract_features(win))
                prediction  = clf.predict(features_scaled)[0]
                probs       = clf.predict_proba(features_scaled)[0]
                max_prob    = np.max(probs)

                # Command mapping
                if max_prob < CONFIDENCE_THRESH:
                    command = "COAST  (low confidence)"
                elif prediction == 'flex':
                    command = "FORWARD"
                elif prediction == 'extend':
                    command = "STOP"
                else:
                    command = "COAST  (rest)"

                print(f"{command:<28} | conf={max_prob:.2f} | "
                      f"flex={probs[idx_flex]:.2f} ext={probs[idx_ext]:.2f} "
                      f"rest={probs[idx_rest]:.2f} | RMS={np.sqrt(np.mean(win**2)):.1f}µV")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopped.")