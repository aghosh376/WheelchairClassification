"""
live_predict_wrapper.py — v3 two-stage cascade, importable prediction stream.

Exposes the same three-function API that run_live_to_serial.py calls:
    load_model()                              → (clf, scaler)
    connect_lsl_inlet()                       → StreamInlet
    live_prediction_stream(inlet, clf, scaler) → generator of result dicts

Internally, "clf" and "scaler" are each tuples that carry both v3 models:
    clf    = (clf_motion, clf_location)
    scaler = (scaler_mot, scaler_loc)

run_live_to_serial.py treats them as opaque objects and passes them straight
through, so the tuple trick keeps the external API identical while letting
live_prediction_stream unpack both models cleanly.

Result dict yielded each window:
    {
        "command"   : str   — "FORWARD" | "STOP" | "COAST" | "STOP (Low Confidence)"
        "raw_label" : str   — compound label e.g. "flex_up", "extend_down", "rest_up"
        "motion"    : str   — "flex" | "extend" | "rest"
        "location"  : str   — "up"   | "down"
        "confidence": float — motion stage max probability (0.0–1.0)
        "rms"       : float — RMS of the winning channel window
    }

Active channel config (2 patches, one arm):
    ACTIVE_CHANNELS maps Ganglion channel index → patch metadata.
    Ch1 = upper forearm (up),  Ch2 = lower forearm (down).
    Change the keys here if your patches are on different Ganglion pins.
"""

import os
import time
import numpy as np
import joblib
from collections import deque
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import butter, iirnotch, lfilter, welch

# ── Paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

_MOTION_CLF_PATH   = os.path.join(_DIR, 'emg_motion_clf_v3.pkl')
_LOCATION_CLF_PATH = os.path.join(_DIR, 'emg_location_clf_v3.pkl')
_SCALER_MOT_PATH   = os.path.join(_DIR, 'emg_scaler_motion_v3.pkl')
_SCALER_LOC_PATH   = os.path.join(_DIR, 'emg_scaler_location_v3.pkl')
_FEATURE_COLS_PATH = os.path.join(_DIR, 'emg_feature_cols.pkl')

# ── Signal config (must match training) ──────────────────────────────────────
FS                  = 200.0
WINDOW_SIZE         = 100
FILTER_BUFFER_SIZE  = 250
SCALE_FACTOR        = 1_000_000.0   # V → µV
CONFIDENCE_THRESHOLD = 0.60

# ── Active channel config ─────────────────────────────────────────────────────
# Only these Ganglion channel indices are classified. Change keys to match
# which pins your EMG patches are physically connected to.
ACTIVE_CHANNELS = {
    1: {'arm': 'R', 'pos': 'upper', 'location': 'up',   'primary': 'flex'},
    2: {'arm': 'R', 'pos': 'lower', 'location': 'down', 'primary': 'extend'},
}
GANGLION_TOTAL_CHANNELS = 4


# ── Filters (built once at import time) ──────────────────────────────────────
_nyq              = 0.5 * FS
_b_band, _a_band  = butter(4, [20.0 / _nyq, 99.0 / _nyq], btype='band')
_b_notch, _a_notch = iirnotch(60.0 / _nyq, 30)


# =============================================================================
# Public API
# =============================================================================

def load_model():
    """
    Load the v3 two-stage cascade models and scalers.

    Returns
    -------
    clf    : tuple (clf_motion, clf_location)
    scaler : tuple (scaler_mot, scaler_loc)

    run_live_to_serial.py passes these straight through to
    live_prediction_stream(), which unpacks them internally.
    """
    missing = [p for p in [_MOTION_CLF_PATH, _LOCATION_CLF_PATH,
                            _SCALER_MOT_PATH, _SCALER_LOC_PATH,
                            _FEATURE_COLS_PATH]
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing v3 pkl files — copy all 5 next to this script:\n" +
            "\n".join(f"  {p}" for p in missing)
        )

    clf_motion   = joblib.load(_MOTION_CLF_PATH)
    clf_location = joblib.load(_LOCATION_CLF_PATH)
    scaler_mot   = joblib.load(_SCALER_MOT_PATH)
    scaler_loc   = joblib.load(_SCALER_LOC_PATH)
    feature_cols = joblib.load(_FEATURE_COLS_PATH)

    print("[v3] Models loaded successfully.")
    print(f"     Motion classes  : {list(clf_motion.classes_)}")
    print(f"     Location classes: {list(clf_location.classes_)}")
    print(f"     Features        : {feature_cols}")
    print(f"     Active channels : {sorted(ACTIVE_CHANNELS.keys())}")

    # Pack as tuples — run_live_to_serial treats them as opaque
    clf    = (clf_motion, clf_location)
    scaler = (scaler_mot, scaler_loc)
    return clf, scaler


def connect_lsl_inlet(target_name='obci_eeg1', target_type='EEG', timeout=5.0):
    """
    Resolve an LSL stream and return a connected StreamInlet.
    Tries by name first, then by type, then lists available streams.
    """
    print("Scanning for LSL stream...")
    streams = resolve_byprop('name', target_name, timeout=timeout)
    if not streams:
        streams = resolve_byprop('type', target_type, timeout=timeout)
    if not streams:
        print("[ERROR] No LSL stream found. Visible streams:")
        for s in resolve_streams(wait_time=2.0):
            print(f"  - {s.name()} | {s.type()} | {s.channel_count()}ch")
        raise RuntimeError(
            "No LSL stream found. Make sure your EMG hardware is streaming."
        )
    inlet = StreamInlet(streams[0])
    info  = streams[0]
    print(f"[LSL] Connected: '{info.name()}' | "
          f"{info.channel_count()} channels @ {info.nominal_srate()} Hz")
    return inlet


def live_prediction_stream(inlet, clf, scaler):
    """
    Generator — yields one result dict per prediction window.

    Parameters
    ----------
    inlet  : StreamInlet  (from connect_lsl_inlet)
    clf    : tuple (clf_motion, clf_location)   (from load_model)
    scaler : tuple (scaler_mot, scaler_loc)     (from load_model)

    Yields
    ------
    dict with keys:
        command    — "FORWARD" | "STOP" | "COAST" | "STOP (Low Confidence)"
        raw_label  — compound e.g. "flex_up", "extend_down", "rest_down"
        motion     — "flex" | "extend" | "rest"
        location   — "up" | "down"
        confidence — float, motion stage probability of winning class
        rms        — float, RMS of the winning channel's window
    """
    clf_motion, clf_location = clf
    scaler_mot, scaler_loc   = scaler

    motion_labels = list(clf_motion.classes_)
    loc_labels    = list(clf_location.classes_)

    idx_flex = motion_labels.index('flex')
    idx_ext  = motion_labels.index('extend')
    idx_rest = motion_labels.index('rest')

    active_ch_ids = sorted(ACTIVE_CHANNELS.keys())   # e.g. [1, 2]

    emg_buffer = deque(maxlen=FILTER_BUFFER_SIZE)

    while True:
        chunk, _ = inlet.pull_chunk()

        if chunk:
            for sample in chunk:
                # Buffer all Ganglion channels for correct filter alignment
                emg_buffer.append([
                    sample[ch] * SCALE_FACTOR
                    for ch in range(GANGLION_TOTAL_CHANNELS)
                ])

            if len(emg_buffer) == FILTER_BUFFER_SIZE:
                raw = np.array(emg_buffer)            # (250, 4)
                raw -= np.mean(raw, axis=0)            # DC removal

                filtered = lfilter(_b_band,  _a_band,  raw, axis=0)
                filtered = lfilter(_b_notch, _a_notch, filtered, axis=0)

                win = filtered[-WINDOW_SIZE:, :]       # (100, 4)

                # ── 8 features for each active channel only ───────────
                features = np.array([
                    _extract_features(win[:, ch])
                    for ch in active_ch_ids
                ])                                     # (n_active, 8)

                # ── Stage 1: motion ───────────────────────────────────
                mot_scaled   = scaler_mot.transform(features)
                motion_probs = clf_motion.predict_proba(mot_scaled)

                conf_flex = motion_probs[:, idx_flex]
                conf_ext  = motion_probs[:, idx_ext]
                conf_rest = motion_probs[:, idx_rest]  # noqa: F841 (kept for completeness)

                # ── Stage 2: location (for raw_label logging) ─────────
                loc_scaled = scaler_loc.transform(features)
                loc_probs  = clf_location.predict_proba(loc_scaled)

                # ── Arbitration: best motion across active channels ───
                best_flex_i    = int(np.argmax(conf_flex))
                best_flex_conf = float(conf_flex[best_flex_i])
                best_ext_i     = int(np.argmax(conf_ext))
                best_ext_conf  = float(conf_ext[best_ext_i])

                if best_flex_conf > best_ext_conf:
                    winning_i    = best_flex_i
                    motion       = 'flex'
                    confidence   = best_flex_conf
                else:
                    winning_i    = best_ext_i
                    motion       = 'extend'
                    confidence   = best_ext_conf

                # If neither clears the threshold, call it rest
                if confidence < CONFIDENCE_THRESHOLD:
                    motion     = 'rest'
                    winning_i  = int(np.argmax(conf_flex))   # pick any channel for RMS

                location  = loc_labels[np.argmax(loc_probs[winning_i])]
                raw_label = f"{motion}_{location}"
                rms       = float(np.sqrt(np.mean(win[:, active_ch_ids[winning_i]] ** 2)))

                # ── Map motion → wheelchair command ───────────────────
                if confidence < CONFIDENCE_THRESHOLD:
                    command = "STOP (Low Confidence)"
                elif motion == 'flex':
                    command = "FORWARD"
                elif motion == 'extend':
                    command = "STOP"
                else:
                    command = "COAST"

                yield {
                    "command"   : command,
                    "raw_label" : raw_label,
                    "motion"    : motion,
                    "location"  : location,
                    "confidence": confidence,
                    "rms"       : rms,
                }

        time.sleep(0.01)


# =============================================================================
# Internal helpers
# =============================================================================

def _extract_features(win, fs=FS):
    """
    Extract 8 features from a 1-D filtered window.
    Order: [rms, mav, zc, wl, mnf, ssc, mdf, pkf] — must match training.
    """
    rms = np.sqrt(np.mean(win ** 2))
    mav = np.mean(np.abs(win))
    zc  = np.sum(np.diff(np.sign(win)) != 0)
    wl  = np.sum(np.abs(np.diff(win)))
    ssc = np.sum(np.diff(np.sign(np.diff(win))) != 0)

    freqs, psd = welch(win, fs=fs, nperseg=len(win))
    psd_sum    = float(np.sum(psd)) or 1e-10
    mnf        = np.sum(freqs * psd) / psd_sum

    cum_psd = np.cumsum(psd)
    mdf_idx = min(np.searchsorted(cum_psd, psd_sum / 2), len(freqs) - 1)
    mdf     = freqs[mdf_idx]
    pkf     = freqs[np.argmax(psd)]

    return np.array([rms, mav, zc, wl, mnf, ssc, mdf, pkf])
