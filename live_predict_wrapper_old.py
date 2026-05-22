"""
live_predict_wrapper.py — importable prediction stream, original untouched.

live_predict.py cannot be imported directly because it runs blocking code at
module level (model loading, LSL connection, infinite loop).  This file
duplicates the prediction logic inside importable functions and a generator.

live_predict.py is never imported or modified.

Exposed API
-----------
load_model()                → (clf, scaler)
connect_lsl_inlet()         → StreamInlet
live_prediction_stream(inlet, clf, scaler)   → generator of dicts:
    {
        "command":    str    — "STOP" | "STOP (Low Confidence)" | "FORWARD" | "REVERSE" | "UNKNOWN"
        "raw_label":  str    — SVM class: "rest" | "flex" | "extend"
        "confidence": float  — max class probability 0.0–1.0
        "rms":        float  — RMS of current window
    }
"""

import time
import os
import numpy as np
import joblib
from collections import deque
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, iirnotch, lfilter, welch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_model_path  = os.path.join(_script_dir, 'emg_svm_model.pkl')
_scaler_path = os.path.join(_script_dir, 'emg_scaler.pkl')

# Must match the values in live_predict.py
_FS                  = 200.0
_WINDOW_SIZE         = 50
_FILTER_BUFFER_SIZE  = 200
_CONFIDENCE_THRESHOLD = 0.75


def load_model():
    """Load the SVM classifier and scaler.  Raises FileNotFoundError if missing."""
    try:
        clf    = joblib.load(_model_path)
        scaler = joblib.load(_scaler_path)
        print("Model and Scaler loaded successfully from:", _script_dir)
        return clf, scaler
    except FileNotFoundError:
        print(
            f"Error: model files not found.\n"
            f"  Expected: {_model_path}\n"
            f"            {_scaler_path}"
        )
        raise


def connect_lsl_inlet():
    """Resolve an LSL EEG/EMG stream and return a connected StreamInlet."""
    print("Looking for an EMG stream...")
    streams = resolve_byprop('type', 'EEG')   # adjust 'type' to match your stream
    inlet = StreamInlet(streams[0])
    print("Connected to stream.")
    return inlet


def live_prediction_stream(inlet, clf, scaler):
    """
    Generator — yields one dict per prediction window.

    Runs the same signal-processing pipeline as live_predict.py:
      bandpass (20–99 Hz) → 60 Hz notch → feature extraction → SVM predict

    Yields
    ------
    dict with keys: command, raw_label, confidence, rms
    """
    nyq = 0.5 * _FS
    b_band, a_band = butter(4, [20.0 / nyq, 99.0 / nyq], btype='band')
    w0 = 60.0 / nyq
    b_notch, a_notch = iirnotch(w0, 30)

    emg_buffer = deque(maxlen=_FILTER_BUFFER_SIZE)

    while True:
        chunk, _ = inlet.pull_chunk()

        if chunk:
            for sample in chunk:
                emg_buffer.append(sample[0])   # single-channel at index 0

            if len(emg_buffer) == _FILTER_BUFFER_SIZE:
                raw_signal = np.array(emg_buffer)

                filtered = lfilter(b_band, a_band, raw_signal)
                filtered = lfilter(b_notch, a_notch, filtered)
                win = filtered[-_WINDOW_SIZE:]

                rms = np.sqrt(np.mean(win ** 2))
                mav = np.mean(np.abs(win))
                zc  = np.sum(np.diff(np.sign(win)) != 0)
                wl  = np.sum(np.abs(np.diff(win)))
                ssc = np.sum(np.diff(np.sign(np.diff(win))) != 0)

                freqs, psd = welch(win, fs=_FS, nperseg=len(win))
                psd_sum = np.sum(psd) or 1e-10
                mnf = np.sum(freqs * psd) / psd_sum

                features        = np.array([[rms, mav, zc, wl, mnf, ssc]])
                features_scaled = scaler.transform(features)

                raw_label   = clf.predict(features_scaled)[0]
                probs       = clf.predict_proba(features_scaled)[0]
                confidence  = float(np.max(probs))

                if confidence < _CONFIDENCE_THRESHOLD:
                    command = "STOP (Low Confidence)"
                elif raw_label == 'rest':
                    command = "STOP"
                elif raw_label == 'flex':
                    command = "FORWARD"
                elif raw_label == 'extend':
                    command = "REVERSE"
                else:
                    command = "UNKNOWN"

                yield {
                    "command":    command,
                    "raw_label":  raw_label,
                    "confidence": confidence,
                    "rms":        float(rms),
                }

        time.sleep(0.01)
