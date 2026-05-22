import time
import numpy as np
import os
import joblib
import serial
import platform
import sys
from collections import deque
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import butter, iirnotch, lfilter, welch

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Helper Function for Robust Ganglion Connection ---
def connect_to_openbci(target_name='obci_eeg1', target_type='EEG', timeout=3.0):
    print(f"Scanning for OpenBCI stream (Name: {target_name} or Type: {target_type})...")
    streams = resolve_byprop('name', target_name, timeout=timeout)
    if not streams:
        streams = resolve_byprop('type', target_type, timeout=timeout)
    if not streams:
        print("\n[ERROR] No matching LSL streams found.")
        all_streams = resolve_streams(wait_time=2.0)
        for s in all_streams:
            print(f"  - Name: {s.name()} | Type: {s.type()}")
        exit()
    inlet = StreamInlet(streams[0])
    print(f"[SUCCESS] Connected to stream: {streams[0].name()} (Type: {streams[0].type()})")
    return inlet

# --- Helper Function for Serial Port Detection (Cross-platform) ---
def find_serial_ports():
    """Detect available serial ports on Mac and Windows."""
    ports = []
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        import glob
        ports = glob.glob('/dev/tty.*') + glob.glob('/dev/cu.*')
    elif system == 'Windows':
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DEVICEMAP\SERIALCOMM')
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                    ports.append(value)
                    i += 1
                except OSError:
                    break
            winreg.CloseKey(key)
        except:
            pass
    
    return ports

def connect_to_wheelchair(port=None, baudrate=9600, timeout=1.0):
    """Connect to wheelchair Arduino via serial port.
    Args:
        port: Serial port name. If None, auto-detects first available port.
        baudrate: Baud rate for serial communication (default 9600).
        timeout: Serial read/write timeout.
    Returns:
        Serial object or None if connection fails.
    """
    if port is None:
        available_ports = find_serial_ports()
        if not available_ports:
            print("[ERROR] No serial ports found. Connect the wheelchair and try again.")
            return None
        port = available_ports[0]
        print(f"[AUTO] Detected serial port: {port}")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"[SUCCESS] Connected to wheelchair on {port} at {baudrate} baud")
        return ser
    except serial.SerialException as e:
        print(f"[ERROR] Failed to connect to {port}: {e}")
        print("        Make sure the wheelchair is connected and powered on.")
        return None

def send_command_to_wheelchair(ser, command):
    """Send command to wheelchair via serial.
    Args:
        ser: Serial object
        command: Single character command ('F', 'S', 'C')
    """
    if ser and ser.is_open:
        try:
            ser.write(command.encode() + b'\n')  # Send command + newline
            ser.flush()
        except serial.SerialException as e:
            print(f"[WARNING] Failed to send command: {e}")

# ==========================================
# 1. Load Pre-trained Model and Scaler
# ==========================================
model_path = os.path.join(script_dir, 'emg_svm_model.pkl')
scaler_path = os.path.join(script_dir, 'emg_scaler.pkl')

print("Loading classifier...")
try:
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded.")
except FileNotFoundError:
    print(f"Error: Files not found at:\n{model_path}")
    exit()

# Map the classes dynamically so we know exactly which column holds which probability
class_labels = list(clf.classes_)
try:
    idx_flex = class_labels.index('flex')
    idx_ext = class_labels.index('extend')
    idx_rest = class_labels.index('rest')
except ValueError:
    print("Error: The model classes do not match 'flex', 'extend', 'rest'. Check your training labels.")
    exit()

# ==========================================
# 2. Setup Signal Processing Parameters
# ==========================================
fs = 200.0
nyq = 0.5 * fs
b_band, a_band = butter(4, [20.0/nyq, 99.0/nyq], btype='band')
w0 = 60.0 / nyq
b_notch, a_notch = iirnotch(w0, 30)

window_size = 100 # Updated to match your new training script
filter_buffer_size = 250 # Slightly larger buffer for stable filtering
emg_buffer = deque(maxlen=filter_buffer_size)

# Adjust this if your live data is in Volts but training was in microVolts
SCALE_FACTOR = 1000000.0 

# Confidence threshold to trigger a movement
THRESHOLD = 0.60

# ========== PREDICTION SMOOTHING PARAMETERS ==========
# Window size for averaging predictions over time (smoothing to avoid twitches)
PREDICTION_WINDOW_SIZE = 10  # Average over last N predictions
prediction_history = deque(maxlen=PREDICTION_WINDOW_SIZE)  # Stores (command, max_conf) tuples 

# ========== SERIAL COMMUNICATION PARAMETERS ==========
WHEELCHAIR_PORT = None  # Auto-detect, or set to specific port (e.g., '/dev/ttyUSB0' or 'COM3')
WHEELCHAIR_BAUDRATE = 9600  # Match your Arduino sketch
WHEELCHAIR_TIMEOUT = 1.0

# ==========================================
# 3. Connect to LSL 
# ==========================================
inlet = connect_to_openbci(target_name='obci_eeg1', target_type='EEG')

# ==========================================
# 3b. Connect to Wheelchair (Serial)
# ==========================================
serialchair = connect_to_wheelchair(port=WHEELCHAIR_PORT, baudrate=WHEELCHAIR_BAUDRATE, timeout=WHEELCHAIR_TIMEOUT)
if serialchair is None:
    print("[WARNING] Wheelchair not connected. Running in SIMULATION mode (no serial output).")
    wheelchair_enabled = False
else:
    wheelchair_enabled = True

# ==========================================
# 4. Live Prediction Loop (Simulation Mode)
# ==========================================
print("Starting live dual-channel prediction... Press Ctrl+C to stop.")
print(f"Prediction smoothing window: {PREDICTION_WINDOW_SIZE} predictions")
print(f"Confidence threshold: {THRESHOLD*100:.0f}%")
if wheelchair_enabled:
    print(f"Wheelchair Status: CONNECTED and READY")
else:
    print(f"Wheelchair Status: DISCONNECTED (simulation mode only)")
print("="*70)
last_state = None 

try:
    while True:
        chunk, timestamps = inlet.pull_chunk()
        
        if chunk:
            for sample in chunk:
                # We pull TWO channels now. 
                # sample[0] = Top Patch (Flex Expert)
                # sample[1] = Bottom Patch (Extend Expert)
                # Adjust indices if you plugged into different Ganglion pins.
                emg_buffer.append([sample[0], sample[1]]) 

            if len(emg_buffer) == filter_buffer_size:
                # Convert to shape (250, 2) and scale
                raw_signal = np.array(emg_buffer) * SCALE_FACTOR
                
                # Kill DC Offset on both channels independently
                raw_signal = raw_signal - np.mean(raw_signal, axis=0)
                
                # Filter across axis 0 (time) for both channels
                filtered = lfilter(b_band, a_band, raw_signal, axis=0)
                filtered = lfilter(b_notch, a_notch, filtered, axis=0)
                
                # Extract the most recent window (100, 2)
                win = filtered[-window_size:, :]
                
                # --- CALCULATE 6 FEATURES FOR BOTH CHANNELS AT ONCE ---
                rms = np.sqrt(np.mean(win**2, axis=0))
                mav = np.mean(np.abs(win), axis=0)
                zc = np.sum(np.diff(np.sign(win), axis=0) != 0, axis=0)
                wl = np.sum(np.abs(np.diff(win, axis=0)), axis=0)
                ssc = np.sum(np.diff(np.sign(np.diff(win, axis=0)), axis=0) != 0, axis=0)
                
                # Mean Frequency (MNF)
                freqs, psd = welch(win, fs=fs, nperseg=len(win), axis=0)
                psd_sum = np.sum(psd, axis=0)
                psd_sum[psd_sum == 0] = 1e-10 
                mnf = np.sum(freqs[:, np.newaxis] * psd, axis=0) / psd_sum
                
                # Combine into shape (2, 6). 
                # Row 0 = Ch1 Features. Row 1 = Ch2 Features.
                features = np.column_stack((rms, mav, zc, wl, mnf, ssc))
                features_scaled = scaler.transform(features)
                
                # Predict probabilities for both channels. Shape = (2, 3)
                probs = clf.predict_proba(features_scaled)
                
                ch1_probs = probs[0] # Top Patch probabilities
                ch2_probs = probs[1] # Bottom Patch probabilities
                
                # --- ARBITRATION LOGIC ---
                # Ch1 looks for Flex. Ch2 looks for Extend.
                conf_flex = ch1_probs[idx_flex]
                conf_ext = ch2_probs[idx_ext]
                
                # Pick the HIGHER confidence for each movement across the two channels
                max_conf_flex = conf_flex
                max_conf_ext = conf_ext
                
                # Decide the winner based on highest confidence
                if max_conf_flex > max_conf_ext and max_conf_flex > THRESHOLD:
                    raw_command = 'F'
                    raw_confidence = max_conf_flex
                elif max_conf_ext > max_conf_flex and max_conf_ext > THRESHOLD:
                    raw_command = 'S'
                    raw_confidence = max_conf_ext
                else:
                    raw_command = 'C'
                    raw_confidence = max(max_conf_flex, max_conf_ext)
                
                # Store in history for smoothing
                prediction_history.append((raw_command, raw_confidence))
                
                # ========== APPLY SMOOTHING WINDOW ==========
                # Count occurrences of each command in the prediction window
                if len(prediction_history) > 0:
                    command_counts = {'F': 0, 'S': 0, 'C': 0}
                    confidence_sums = {'F': 0.0, 'S': 0.0, 'C': 0.0}
                    
                    for cmd, conf in prediction_history:
                        command_counts[cmd] += 1
                        confidence_sums[cmd] += conf
                    
                    # Majority voting with confidence weighting
                    max_count = max(command_counts.values())
                    candidates = [cmd for cmd, count in command_counts.items() if count == max_count]
                    
                    # If tie, pick the one with highest average confidence
                    smoothed_command = max(candidates, key=lambda cmd: confidence_sums[cmd] / max(command_counts[cmd], 1))
                    smoothed_confidence = confidence_sums[smoothed_command] / command_counts[smoothed_command]
                    
                    # Map smoothed command to readable string
                    command_map = {'F': 'FORWARD (Flex)', 'S': 'STOP (Extend)', 'C': 'COAST (No Input)'}
                    command_str = command_map[smoothed_command]
                    current_state = smoothed_command
                else:
                    current_state = raw_command
                    smoothed_confidence = raw_confidence
                    command_str = {'F': 'FORWARD (Flex)', 'S': 'STOP (Extend)', 'C': 'COAST (No Input)'}[raw_command]

                # --- CONSOLE OUTPUT ---
                # Always print stream info to see real-time classification
                print(f"\n{'='*70}")
                print(f"[PREDICTION WINDOW {len(prediction_history)}/{PREDICTION_WINDOW_SIZE}]")
                print(f"  Ch1 (Flex):   {conf_flex*100:5.1f}% | Ch2 (Extend): {conf_ext*100:5.1f}%")
                print(f"  Raw Command:  [{raw_command}] {{'F': 'FORWARD', 'S': 'STOP', 'C': 'COAST'}[raw_command]} (conf: {raw_confidence*100:.1f}%)")
                
                if current_state != last_state and len(prediction_history) == PREDICTION_WINDOW_SIZE:
                    print(f"  >>> SMOOTHED:  [{current_state}] {command_str} (avg_conf: {smoothed_confidence*100:.1f}%)")
                    print(f"      Vote Count: F={command_counts['F']} | S={command_counts['S']} | C={command_counts['C']}")
                    
                    # Send to wheelchair if connected
                    if wheelchair_enabled:
                        send_command_to_wheelchair(serialchair, current_state)
                        print(f"      >>> SENT to wheelchair: '{current_state}'")
                    
                    last_state = current_state

        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\nPrediction loop stopped by user. Exiting safely.")
finally:
    # Clean up serial connection
    if wheelchair_enabled and serialchair and serialchair.is_open:
        serialchair.close()
        print("[CLEANUP] Serial connection closed.")
