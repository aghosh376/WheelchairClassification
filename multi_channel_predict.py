import time
import numpy as np
import os
import joblib
import serial
from collections import deque
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import butter, iirnotch, lfilter, welch

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 0. Setup Serial Connection to Microcontroller
# ==========================================
SERIAL_PORT = 'COM3' # Change to your COM port
BAUD_RATE = 115200

print(f"Connecting to microcontroller on {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print(f"[SUCCESS] Serial connected.")
except serial.SerialException:
    print(f"[WARNING] Could not connect to {SERIAL_PORT}.")
    print("Running in simulation mode (no serial output).")
    ser = None

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

# ==========================================
# 3. Connect to LSL 
# ==========================================
inlet = connect_to_openbci(target_name='obci_eeg1', target_type='EEG')

# ==========================================
# 4. Live Prediction & Serial Loop
# ==========================================
print("Starting live dual-channel prediction... Press Ctrl+C to stop.")
last_sent_command = None 

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
                
                # Decide the winner based on thresholds
                if conf_flex > THRESHOLD and conf_flex > conf_ext:
                    command_str = "FORWARD (Flex)"
                    serial_byte = b'F'
                elif conf_ext > THRESHOLD and conf_ext > conf_flex:
                    command_str = "REVERSE (Extend)"
                    serial_byte = b'R'
                else:
                    command_str = "STOP (Rest/Low Conf)"
                    serial_byte = b'S'

                # --- SEND SERIAL DATA ---
                if serial_byte != last_sent_command:
                    if ser is not None:
                        ser.write(serial_byte)
                    
                    print(f"\n--> SENT TO BOARD: {serial_byte.decode('utf-8')}")
                    print(f"    Action : {command_str}")
                    print(f"    Top Pad   (Flex Conf)  : {conf_flex*100:.0f}%")
                    print(f"    Bottom Pad(Extend Conf): {conf_ext*100:.0f}%")
                    
                    last_sent_command = serial_byte

        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\nPrediction loop stopped by user.")
finally:
    if 'ser' in locals() and ser is not None and ser.is_open:
        ser.write(b'S') 
        ser.close()
        print("Serial port closed safely.")