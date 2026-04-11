import time
import numpy as np
import os
import joblib
from collections import deque
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import butter, iirnotch, lfilter

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Helper Function for Robust Ganglion Connection ---
def connect_to_openbci(target_name='obci_eeg1', target_type='EEG', timeout=3.0):
    print(f"Scanning for OpenBCI stream (Name: {target_name} or Type: {target_type})...")
    
    # 1. Try finding it by its specific OpenBCI name first
    streams = resolve_byprop('name', target_name, timeout=timeout)
    
    # 2. Fallback to generic 'EEG' or 'EMG' type
    if not streams:
        print(f"Stream '{target_name}' not found. Falling back to type '{target_type}'...")
        streams = resolve_byprop('type', target_type, timeout=timeout)
        
    # 3. Error handling: List what IS available if we fail
    if not streams:
        print("\n[ERROR] No matching LSL streams found.")
        print("Make sure the OpenBCI GUI is running and the LSL widget is streaming.")
        print("Currently available streams on your network:")
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
start_time = time.time()
try:
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model and Scaler loaded in {time.time() - start_time:.3f} seconds from:\n{script_dir}")
except FileNotFoundError:
    print(f"Error: Files not found. Python is looking exactly here:\n{model_path}\n{scaler_path}")
    exit()

# ==========================================
# 2. Setup Signal Processing Parameters
# ==========================================
fs = 200.0
nyq = 0.5 * fs
b_band, a_band = butter(4, [20.0/nyq, 99.0/nyq], btype='band')
w0 = 60.0 / nyq
b_notch, a_notch = iirnotch(w0, 30)

window_size = 50 
filter_buffer_size = 200 
emg_buffer = deque(maxlen=filter_buffer_size)

# ==========================================
# 3. Connect to LSL (Using robust helper)
# ==========================================
# In the OpenBCI GUI, the LSL widget usually names the stream 'obci_eeg1'. 
# If you changed it to EMG in the GUI, change target_type to 'EMG'.
inlet = connect_to_openbci(target_name='obci_eeg1', target_type='EEG')

# ==========================================
# 4. Live Prediction Loop
# ==========================================
print("Starting live prediction... Press Ctrl+C to stop.")
try:
    while True:
        chunk, timestamps = inlet.pull_chunk()
        
        if chunk:
            # Ganglion sends 4 channels. 
            # If you only use Channel 1, index [0]. Update if you use a different channel.
            for sample in chunk:
                emg_buffer.append(sample[0]) 

            if len(emg_buffer) == filter_buffer_size:
                raw_signal = np.array(emg_buffer)
                
                filtered = lfilter(b_band, a_band, raw_signal)
                filtered = lfilter(b_notch, a_notch, filtered)
                win = filtered[-window_size:]
                
                rms = np.sqrt(np.mean(win**2))
                mav = np.mean(np.abs(win))
                zc = np.sum(np.diff(np.sign(win)) != 0)
                wl = np.sum(np.abs(np.diff(win)))
                
                features = np.array([[rms, mav, zc, wl]])
                features_scaled = scaler.transform(features)
                
                prediction = clf.predict(features_scaled)[0]
                probabilities = clf.predict_proba(features_scaled)[0]
                max_prob = np.max(probabilities)
                
                if max_prob < 0.75:
                    command = "STOP (Low Confidence)"
                elif prediction == 'rest':
                    command = "STOP"
                elif prediction == 'flex':
                    command = "FORWARD"
                elif prediction == 'extend':
                    command = "REVERSE"
                else:
                    command = "UNKNOWN"
                    
                print(f"Command: {command:<25} | Confidence: {max_prob:.2f} | RMS: {rms:.2f}")

        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\nPrediction loop stopped by user.")