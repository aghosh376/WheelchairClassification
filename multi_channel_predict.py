"""
multi_channel_predict.py — 4-channel EMG wheelchair controller.

Architecture:
  - Same single-channel SVM model run independently on each of 4 channels.
  - Each channel produces its own [extend, flex, rest] probability vector.
  - Arbitration layer picks the highest-confidence command across all channels.
  - Smoothing window (majority vote) prevents twitchy commands.
  - Serial output to Arduino wheelchair controller.

Channel layout (adjust CHANNEL_ROLES to match your wiring):
  Ch0 = sample[0] → Right arm, upper patch  (primary flex)
  Ch1 = sample[1] → Right arm, lower patch  (primary extend)
  Ch2 = sample[2] → Left arm,  upper patch  (secondary flex)
  Ch3 = sample[3] → Left arm,  lower patch  (secondary extend)
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
# CONFIGURATION — edit these to tune behaviour
# ══════════════════════════════════════════════════════════════
NUM_CHANNELS          = 4        # Ganglion has 4 channels
CHANNEL_ROLES = {
    0: {'arm': 'R', 'pos': 'upper', 'primary': 'flex'},
    1: {'arm': 'R', 'pos': 'lower', 'primary': 'extend'},
    2: {'arm': 'L', 'pos': 'upper', 'primary': 'flex'},
    3: {'arm': 'L', 'pos': 'lower', 'primary': 'extend'},
}

FS                    = 200.0
WINDOW_SIZE           = 100      # must match training
FILTER_BUFFER_SIZE    = 250
SCALE_FACTOR          = 1_000_000.0   # V → µV

CONFIDENCE_THRESHOLD  = 0.60     # minimum confidence to accept a command
PREDICTION_WINDOW     = 10       # smoothing: majority vote over last N raw predictions

WHEELCHAIR_PORT       = None     # None = auto-detect; or set e.g. 'COM3' / '/dev/ttyUSB0'
WHEELCHAIR_BAUDRATE   = 9600


# ══════════════════════════════════════════════════════════════
# 1. Load model
# ══════════════════════════════════════════════════════════════
print("Loading classifier...")
try:
    clf    = joblib.load(os.path.join(script_dir, 'emg_svm_model.pkl'))
    scaler = joblib.load(os.path.join(script_dir, 'emg_scaler.pkl'))
    print(f"[OK] Classes: {clf.classes_} | Features expected: {clf.n_features_in_}")
except FileNotFoundError:
    print(f"[ERROR] .pkl files not found in {script_dir}")
    exit()

class_labels = list(clf.classes_)
try:
    idx_flex = class_labels.index('flex')
    idx_ext  = class_labels.index('extend')
    idx_rest = class_labels.index('rest')
except ValueError as e:
    print(f"[ERROR] Unexpected class labels: {clf.classes_}. Expected flex/extend/rest.")
    exit()


# ══════════════════════════════════════════════════════════════
# 2. Signal processing setup
# ══════════════════════════════════════════════════════════════
nyq              = 0.5 * FS
b_band,  a_band  = butter(4, [20.0/nyq, 99.0/nyq], btype='band')
b_notch, a_notch = iirnotch(60.0/nyq, 30)

emg_buffer       = deque(maxlen=FILTER_BUFFER_SIZE)  # stores (N, 4) rows
prediction_history = deque(maxlen=PREDICTION_WINDOW)


# ══════════════════════════════════════════════════════════════
# 3. Feature extraction — must be identical to training
# ══════════════════════════════════════════════════════════════
def extract_features_single(win, fs=200.0):
    """
    Extract 6 features from a 1D window of length WINDOW_SIZE.
    Feature order: [rms, mav, zc, wl, mnf, ssc] — must match training scaler.
    """
    rms = np.sqrt(np.mean(win**2))
    mav = np.mean(np.abs(win))
    zc  = np.sum(np.diff(np.sign(win)) != 0)
    wl  = np.sum(np.abs(np.diff(win)))
    ssc = np.sum(np.diff(np.sign(np.diff(win))) != 0)
    freqs, psd = welch(win, fs=fs, nperseg=len(win))
    psd_sum = np.sum(psd) or 1e-10
    mnf = np.sum(freqs * psd) / psd_sum
    return np.array([rms, mav, zc, wl, mnf, ssc])  # shape: (6,)


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
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DEVICEMAP\SERIALCOMM')
            i = 0
            while True:
                try:
                    _, value, _ = winreg.EnumValue(key, i)
                    ports.append(value); i += 1
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
    print(f"Scanning for LSL stream...")
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
    print(f"[OK] Connected: '{info.name()}' | {info.channel_count()} channels @ {info.nominal_srate()} Hz")
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
    counts  = {'F': 0, 'S': 0, 'C': 0}
    conf_sum = {'F': 0.0, 'S': 0.0, 'C': 0.0}
    for cmd, conf in history:
        counts[cmd]   += 1
        conf_sum[cmd] += conf

    max_count  = max(counts.values())
    candidates = [c for c, n in counts.items() if n == max_count]
    winner     = max(candidates, key=lambda c: conf_sum[c] / max(counts[c], 1))
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
# 8. Live prediction loop
# ══════════════════════════════════════════════════════════════
CMD_LABELS = {'F': 'FORWARD', 'S': 'STOP', 'C': 'COAST'}
last_sent  = None

print(f"\n{'='*70}")
print(f"4-Channel EMG Wheelchair Controller")
print(f"  Channels  : {NUM_CHANNELS}  |  Window: {WINDOW_SIZE} samples  |  Threshold: {CONFIDENCE_THRESHOLD:.0%}")
print(f"  Smoothing : last {PREDICTION_WINDOW} predictions (majority vote)")
print(f"  Wheelchair: {'CONNECTED' if wheelchair_enabled else 'SIMULATION MODE'}")
print(f"{'='*70}\n")

try:
    while True:
        chunk, _ = inlet.pull_chunk()

        if chunk:
            for sample in chunk:
                # Store all 4 channels per sample — shape of each row: (4,)
                emg_buffer.append([
                    sample[0] * SCALE_FACTOR,
                    sample[1] * SCALE_FACTOR,
                    sample[2] * SCALE_FACTOR,
                    sample[3] * SCALE_FACTOR,
                ])

            if len(emg_buffer) == FILTER_BUFFER_SIZE:
                raw = np.array(emg_buffer)          # shape: (250, 4)
                raw -= np.mean(raw, axis=0)          # per-channel DC removal

                # Filter all 4 channels along time axis
                filtered = lfilter(b_band,  a_band,  raw, axis=0)
                filtered = lfilter(b_notch, a_notch, filtered, axis=0)

                win = filtered[-WINDOW_SIZE:, :]     # shape: (100, 4)

                # ── Extract 6 features per channel → shape (4, 6) ──────────
                features = np.array([
                    extract_features_single(win[:, ch], fs=FS)
                    for ch in range(NUM_CHANNELS)
                ])                                   # shape: (4, 6)

                # Scale using the same scaler as training (treats rows as samples)
                features_scaled = scaler.transform(features)   # shape: (4, 6)

                # ── Classify each channel independently ──────────────────
                probs = clf.predict_proba(features_scaled)      # shape: (4, 3)
                # probs[ch] = [p_extend, p_flex, p_rest]

                conf_flex = probs[:, idx_flex]   # shape: (4,) — one per channel
                conf_ext  = probs[:, idx_ext]    # shape: (4,)
                conf_rest = probs[:, idx_rest]   # shape: (4,)

                # ── Arbitration: best flex and best extend across ALL 4 ch ─
                best_flex_conf = np.max(conf_flex)
                best_flex_ch   = int(np.argmax(conf_flex))
                best_ext_conf  = np.max(conf_ext)
                best_ext_ch    = int(np.argmax(conf_ext))

                if best_flex_conf > best_ext_conf and best_flex_conf > CONFIDENCE_THRESHOLD:
                    raw_cmd  = 'F'
                    raw_conf = best_flex_conf
                    win_ch   = best_flex_ch
                elif best_ext_conf > best_flex_conf and best_ext_conf > CONFIDENCE_THRESHOLD:
                    raw_cmd  = 'S'
                    raw_conf = best_ext_conf
                    win_ch   = best_ext_ch
                else:
                    raw_cmd  = 'C'
                    raw_conf = max(best_flex_conf, best_ext_conf)
                    win_ch   = -1

                prediction_history.append((raw_cmd, raw_conf))

                # ── Smoothing ─────────────────────────────────────────────
                smoothed_cmd, smoothed_conf, vote_counts = smooth_predictions(prediction_history)

                # ── Console output — always show per-channel breakdown ────
                print(f"\n{'='*70}")
                print(f"[WINDOW {len(prediction_history)}/{PREDICTION_WINDOW}]")
                for ch in range(NUM_CHANNELS):
                    role   = CHANNEL_ROLES[ch]
                    marker = ' ◄ winner' if ch == win_ch else ''
                    print(f"  Ch{ch} ({role['arm']} {role['pos']:5s}): "
                          f"flex={conf_flex[ch]*100:5.1f}%  "
                          f"ext={conf_ext[ch]*100:5.1f}%  "
                          f"rest={conf_rest[ch]*100:5.1f}%{marker}")

                print(f"  Raw:      [{raw_cmd}] {CMD_LABELS[raw_cmd]} "
                      f"(conf={raw_conf*100:.1f}%)")

                # Only act + print smoothed result when window is full and state changes
                if (len(prediction_history) == PREDICTION_WINDOW
                        and smoothed_cmd != last_sent):
                    print(f"  SMOOTHED: [{smoothed_cmd}] {CMD_LABELS[smoothed_cmd]} "
                          f"(avg={smoothed_conf*100:.1f}%) "
                          f"votes F={vote_counts['F']} S={vote_counts['S']} C={vote_counts['C']}")

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