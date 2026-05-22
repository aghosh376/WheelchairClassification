"""
multi_channel_predict.py — 2-channel EMG wheelchair controller (v3 cascade).

Architecture:
  - Two-stage cascade classifier trained offline:
      Stage 1 (motion):   flex / extend / rest    — emg_motion_clf_v3.pkl
      Stage 2 (location): up / down               — emg_location_clf_v3.pkl
  - Only the 2 active EMG channels are classified. The Ganglion sends 4
    channels over LSL but unused channels (noise/ground) are ignored.
  - Arbitration picks the highest-confidence MOTION command across the
    active channels. Location is logged for debug only.
  - Majority-vote smoothing over last N predictions prevents jitter.
  - Serial output to Arduino wheelchair controller.

Wiring (edit ACTIVE_CHANNELS to match your setup):
  Ganglion Ch1 → upper forearm patch (up)    — primary flex signal
  Ganglion Ch2 → lower forearm patch (down)  — primary extend signal
  Ch0, Ch3     → unused / not connected

CRITICAL — feature order must match training:
  v3 uses 8 features: [rms, mav, zc, wl, mnf, ssc, mdf, pkf]
"""

import time
import numpy as np
import os
import joblib
import serial
import platform
from collections import deque
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import butter, iirnotch, lfilter, welch

script_dir = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
# The Ganglion sends 4 channels but you only have 2 EMG patches.
# ACTIVE_CHANNELS maps Ganglion channel index → patch info.
# Change the keys to match which Ganglion pins your patches are on.
#   Channel 1 = upper forearm patch (up)   → trained as location 'up'
#   Channel 2 = lower forearm patch (down) → trained as location 'down'
# Channels 0 and 3 are unused (noise) and will be ignored entirely.
ACTIVE_CHANNELS = {
    1: {'arm': 'R', 'pos': 'upper', 'location': 'up',   'primary': 'flex'},
    2: {'arm': 'R', 'pos': 'lower', 'location': 'down', 'primary': 'extend'},
}
GANGLION_TOTAL_CHANNELS = 4      # Ganglion always sends 4 in the LSL stream

FS                 = 200.0
WINDOW_SIZE        = 100       # must match training
FILTER_BUFFER_SIZE = 250
SCALE_FACTOR       = 1_000_000.0   # V → µV

CONFIDENCE_THRESHOLD = 0.60    # min confidence to accept a motion command
PREDICTION_WINDOW    = 10      # majority-vote smoothing depth

WHEELCHAIR_PORT      = None    # None = auto-detect
WHEELCHAIR_BAUDRATE  = 9600


# ══════════════════════════════════════════════════════════════
# 1. Load v3 two-stage cascade models
# ══════════════════════════════════════════════════════════════
print("Loading v3 cascade classifiers...")
try:
    clf_motion   = joblib.load(os.path.join(script_dir, 'emg_motion_clf_v3.pkl'))
    clf_location = joblib.load(os.path.join(script_dir, 'emg_location_clf_v3.pkl'))
    scaler_mot   = joblib.load(os.path.join(script_dir, 'emg_scaler_motion_v3.pkl'))
    scaler_loc   = joblib.load(os.path.join(script_dir, 'emg_scaler_location_v3.pkl'))
    feature_cols = joblib.load(os.path.join(script_dir, 'emg_feature_cols.pkl'))
    print(f"[OK] Motion classes:   {list(clf_motion.classes_)}")
    print(f"[OK] Location classes: {list(clf_location.classes_)}")
    print(f"[OK] Feature order:    {feature_cols}")
except FileNotFoundError as e:
    print(f"[ERROR] pkl file not found: {e}")
    print(f"  Expected in: {script_dir}")
    print(f"  Required files:")
    print(f"    emg_motion_clf_v3.pkl")
    print(f"    emg_location_clf_v3.pkl")
    print(f"    emg_scaler_motion_v3.pkl")
    print(f"    emg_scaler_location_v3.pkl")
    print(f"    emg_feature_cols.pkl")
    exit()

# Build index lookups for motion classes
motion_labels = list(clf_motion.classes_)
try:
    idx_flex = motion_labels.index('flex')
    idx_ext  = motion_labels.index('extend')
    idx_rest = motion_labels.index('rest')
except ValueError:
    print(f"[ERROR] Unexpected motion classes: {motion_labels}")
    print(f"  Expected: ['extend', 'flex', 'rest']")
    exit()

# Build index lookups for location classes
loc_labels = list(clf_location.classes_)
try:
    idx_up   = loc_labels.index('up')
    idx_down = loc_labels.index('down')
except ValueError:
    print(f"[ERROR] Unexpected location classes: {loc_labels}")
    print(f"  Expected: ['down', 'up']")
    exit()


# ══════════════════════════════════════════════════════════════
# 2. Signal processing setup
# ══════════════════════════════════════════════════════════════
nyq              = 0.5 * FS
b_band,  a_band  = butter(4, [20.0 / nyq, 99.0 / nyq], btype='band')
b_notch, a_notch = iirnotch(60.0 / nyq, 30)

emg_buffer         = deque(maxlen=FILTER_BUFFER_SIZE)
prediction_history = deque(maxlen=PREDICTION_WINDOW)


# ══════════════════════════════════════════════════════════════
# 3. Feature extraction — 8 features, must match v3 training
#    Order: [rms, mav, zc, wl, mnf, ssc, mdf, pkf]
# ══════════════════════════════════════════════════════════════
def extract_features_single(win, fs=FS):
    """
    Extract 8 features from a 1-D filtered window.
    Returns shape (8,) in the order matching emg_feature_cols.pkl:
      rms, mav, zc, wl, mnf, ssc, mdf, pkf
    """
    rms = np.sqrt(np.mean(win ** 2))
    mav = np.mean(np.abs(win))
    zc  = np.sum(np.diff(np.sign(win)) != 0)
    wl  = np.sum(np.abs(np.diff(win)))
    ssc = np.sum(np.diff(np.sign(np.diff(win))) != 0)

    freqs, psd = welch(win, fs=fs, nperseg=len(win))
    psd_sum = np.sum(psd) or 1e-10

    mnf = np.sum(freqs * psd) / psd_sum

    # Median frequency: frequency at which cumulative PSD reaches 50%
    cum_psd = np.cumsum(psd)
    mdf_idx = min(np.searchsorted(cum_psd, psd_sum / 2), len(freqs) - 1)
    mdf = freqs[mdf_idx]

    # Peak frequency: frequency with highest PSD
    pkf = freqs[np.argmax(psd)]

    return np.array([rms, mav, zc, wl, mnf, ssc, mdf, pkf])


# ══════════════════════════════════════════════════════════════
# 4. Serial / wheelchair helpers
# ══════════════════════════════════════════════════════════════
def find_serial_ports():
    system = platform.system()
    if system == 'Darwin':
        import glob
        return glob.glob('/dev/tty.*') + glob.glob('/dev/cu.*')
    elif system == 'Windows':
        import winreg
        ports = []
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DEVICEMAP\SERIALCOMM')
            i = 0
            while True:
                try:
                    _, value, _ = winreg.EnumValue(key, i)
                    ports.append(value)
                    i += 1
                except OSError:
                    break
            winreg.CloseKey(key)
        except Exception:
            pass
        return ports
    return []


def connect_wheelchair(port=None, baudrate=9600):
    if port is None:
        ports = find_serial_ports()
        if not ports:
            print("[WARNING] No serial ports found.")
            return None
        port = ports[0]
        print(f"[AUTO] Using serial port: {port}")
    try:
        ser = serial.Serial(port, baudrate, timeout=1.0)
        time.sleep(2)
        print(f"[OK] Wheelchair connected on {port}")
        return ser
    except serial.SerialException as e:
        print(f"[ERROR] Serial connection failed: {e}")
        return None


def send_command(ser, command):
    """Send a single-char command ('F', 'S', 'C') to the Arduino."""
    if ser and ser.is_open:
        try:
            ser.write(command.encode() + b'\n')
            ser.flush()
        except serial.SerialException as e:
            print(f"[WARNING] Serial write failed: {e}")


# ══════════════════════════════════════════════════════════════
# 5. LSL connection
# ══════════════════════════════════════════════════════════════
def connect_lsl(target_name='obci_eeg1', target_type='EEG', timeout=5.0):
    print("Scanning for LSL stream...")
    streams = resolve_byprop('name', target_name, timeout=timeout)
    if not streams:
        streams = resolve_byprop('type', target_type, timeout=timeout)
    if not streams:
        print("[ERROR] No LSL stream found. Visible streams:")
        for s in resolve_streams(wait_time=2.0):
            print(f"  - {s.name()} | {s.type()} | {s.channel_count()}ch")
        exit()
    inlet = StreamInlet(streams[0])
    info  = streams[0]
    print(f"[OK] Connected: '{info.name()}' | {info.channel_count()} channels "
          f"@ {info.nominal_srate()} Hz")
    return inlet


# ══════════════════════════════════════════════════════════════
# 6. Smoothing helper
# ══════════════════════════════════════════════════════════════
def smooth_predictions(history):
    """
    Majority vote over prediction_history.
    Ties broken by highest average confidence.
    Returns (smoothed_command, avg_confidence, vote_counts).
    """
    counts   = {'F': 0, 'S': 0, 'C': 0}
    conf_sum = {'F': 0.0, 'S': 0.0, 'C': 0.0}
    for cmd, conf in history:
        counts[cmd]   += 1
        conf_sum[cmd] += conf

    max_count  = max(counts.values())
    candidates = [c for c, n in counts.items() if n == max_count]
    winner     = max(candidates,
                     key=lambda c: conf_sum[c] / max(counts[c], 1))
    avg_conf   = conf_sum[winner] / counts[winner]
    return winner, avg_conf, counts


# ══════════════════════════════════════════════════════════════
# 7. Connect to hardware
# ══════════════════════════════════════════════════════════════
inlet = connect_lsl()

ser = connect_wheelchair(port=WHEELCHAIR_PORT, baudrate=WHEELCHAIR_BAUDRATE)
wheelchair_enabled = ser is not None
if not wheelchair_enabled:
    print("[INFO] Running in SIMULATION mode — no serial output.")


# ══════════════════════════════════════════════════════════════
# 8. Live prediction loop — v3 two-stage cascade
#
# For each prediction window:
#   1. Extract 8 features from each of the 4 channels  → (4, 8)
#   2. Stage 1: scale with motion scaler, predict_proba → (4, 3) motion probs
#   3. Stage 2: scale with location scaler, predict_proba → (4, 2) location probs
#   4. Arbitration on MOTION probs only:
#        flex  → FORWARD ('F')
#        extend → STOP   ('S')
#        rest   → COAST  ('C')
#   5. Majority-vote smoothing → send to wheelchair
#
# Location predictions are logged for debugging but do NOT
# affect the wheelchair command. If you want location-gated
# arbitration (e.g. only trust a flex prediction if location
# matches the channel's expected patch), uncomment the
# LOCATION GATE block below.
# ══════════════════════════════════════════════════════════════
CMD_LABELS    = {'F': 'FORWARD', 'S': 'STOP', 'C': 'COAST'}
last_sent     = None
active_ch_ids = sorted(ACTIVE_CHANNELS.keys())   # e.g. [1, 2]
n_active      = len(active_ch_ids)

print(f"\n{'=' * 70}")
print(f"2-Channel EMG Wheelchair Controller  (v3 cascade)")
print(f"  Active ch : {active_ch_ids}  (of {GANGLION_TOTAL_CHANNELS} Ganglion channels)")
for ch_id in active_ch_ids:
    role = ACTIVE_CHANNELS[ch_id]
    print(f"    Ch{ch_id} → {role['arm']} arm, {role['pos']} patch "
          f"(location={role['location']}, primary={role['primary']})")
print(f"  Window    : {WINDOW_SIZE} samples  |  Features: "
      f"{len(feature_cols)} ({', '.join(feature_cols)})")
print(f"  Threshold : {CONFIDENCE_THRESHOLD:.0%}")
print(f"  Smoothing : last {PREDICTION_WINDOW} predictions (majority vote)")
print(f"  Wheelchair: {'CONNECTED' if wheelchair_enabled else 'SIMULATION MODE'}")
print(f"{'=' * 70}\n")

try:
    while True:
        chunk, _ = inlet.pull_chunk()

        if chunk:
            for sample in chunk:
                # Buffer ALL Ganglion channels (need them for filtering alignment)
                emg_buffer.append([
                    sample[ch] * SCALE_FACTOR
                    for ch in range(GANGLION_TOTAL_CHANNELS)
                ])

            if len(emg_buffer) == FILTER_BUFFER_SIZE:
                raw = np.array(emg_buffer)               # (250, 4)
                raw -= np.mean(raw, axis=0)               # DC removal per channel

                # Filter all channels (cheap, and keeps indices aligned)
                filtered = lfilter(b_band,  a_band,  raw, axis=0)
                filtered = lfilter(b_notch, a_notch, filtered, axis=0)

                win = filtered[-WINDOW_SIZE:, :]          # (100, 4)

                # ── Extract 8 features ONLY from active channels ─────────
                features = np.array([
                    extract_features_single(win[:, ch], fs=FS)
                    for ch in active_ch_ids
                ])                                        # (n_active, 8)

                # ── Stage 1: motion classification ───────────────────────
                feat_mot_scaled = scaler_mot.transform(features)
                motion_probs    = clf_motion.predict_proba(feat_mot_scaled)  # (n_active, 3)

                conf_flex = motion_probs[:, idx_flex]     # (n_active,)
                conf_ext  = motion_probs[:, idx_ext]
                conf_rest = motion_probs[:, idx_rest]

                # ── Stage 2: location classification (debug/logging) ─────
                feat_loc_scaled = scaler_loc.transform(features)
                loc_probs       = clf_location.predict_proba(feat_loc_scaled)

                loc_pred = [loc_labels[np.argmax(loc_probs[i])]
                            for i in range(n_active)]
                loc_conf = [float(np.max(loc_probs[i]))
                            for i in range(n_active)]

                # ── Arbitration on motion (active channels only) ─────────
                best_flex_idx  = int(np.argmax(conf_flex))   # index into active list
                best_flex_conf = conf_flex[best_flex_idx]
                best_flex_ch   = active_ch_ids[best_flex_idx] # actual Ganglion ch

                best_ext_idx   = int(np.argmax(conf_ext))
                best_ext_conf  = conf_ext[best_ext_idx]
                best_ext_ch    = active_ch_ids[best_ext_idx]

                if (best_flex_conf > best_ext_conf
                        and best_flex_conf > CONFIDENCE_THRESHOLD):
                    raw_cmd  = 'F'
                    raw_conf = best_flex_conf
                    win_ch   = best_flex_ch
                elif (best_ext_conf > best_flex_conf
                      and best_ext_conf > CONFIDENCE_THRESHOLD):
                    raw_cmd  = 'S'
                    raw_conf = best_ext_conf
                    win_ch   = best_ext_ch
                else:
                    raw_cmd  = 'C'
                    raw_conf = float(np.max(conf_rest))
                    win_ch   = -1

                prediction_history.append((raw_cmd, raw_conf))

                # ── Smoothing ────────────────────────────────────────────
                smoothed_cmd, smoothed_conf, vote_counts = \
                    smooth_predictions(prediction_history)

                # ── Console output (active channels only) ────────────────
                print(f"\n{'=' * 70}")
                print(f"[WINDOW {len(prediction_history)}/{PREDICTION_WINDOW}]")
                for i, ch in enumerate(active_ch_ids):
                    role   = ACTIVE_CHANNELS[ch]
                    marker = ' ◄ winner' if ch == win_ch else ''
                    motion_best     = motion_labels[np.argmax(motion_probs[i])]
                    motion_best_pct = float(np.max(motion_probs[i])) * 100
                    print(
                        f"  Ch{ch} ({role['arm']} {role['pos']:5s}): "
                        f"flex={conf_flex[i]*100:5.1f}%  "
                        f"ext={conf_ext[i]*100:5.1f}%  "
                        f"rest={conf_rest[i]*100:5.1f}%  "
                        f"│ loc={loc_pred[i]:4s} ({loc_conf[i]*100:4.1f}%)  "
                        f"→ {motion_best}_{loc_pred[i]} "
                        f"({motion_best_pct:.0f}%){marker}"
                    )

                print(f"  Raw:      [{raw_cmd}] {CMD_LABELS[raw_cmd]} "
                      f"(conf={raw_conf * 100:.1f}%)")

                if (len(prediction_history) == PREDICTION_WINDOW
                        and smoothed_cmd != last_sent):
                    print(
                        f"  SMOOTHED: [{smoothed_cmd}] "
                        f"{CMD_LABELS[smoothed_cmd]} "
                        f"(avg={smoothed_conf * 100:.1f}%) "
                        f"votes F={vote_counts['F']} "
                        f"S={vote_counts['S']} "
                        f"C={vote_counts['C']}"
                    )

                    if wheelchair_enabled:
                        send_command(ser, smoothed_cmd)
                        print(f"  >>> SENT '{smoothed_cmd}' to wheelchair")

                    last_sent = smoothed_cmd

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
finally:
    if wheelchair_enabled and ser and ser.is_open:
        ser.close()
        print("[CLEANUP] Serial connection closed.")
