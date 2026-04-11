import time
import numpy as np
import os
import joblib
from collections import deque
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, iirnotch, lfilter

script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Build the full, absolute paths to the files
model_path = os.path.join(script_dir, 'emg_svm_model.pkl')
scaler_path = os.path.join(script_dir, 'emg_scaler.pkl')

# 3. Load using the absolute paths
try:
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully from:", script_dir)
except FileNotFoundError:
    print(f"Error: Files not found. Python is looking exactly here:\n{model_path}\n{scaler_path}")
    exit()
# 1. Load Pre-trained Model and Scaler
#try:
#    clf = joblib.load('emg_svm_model.pkl')
#    scaler = joblib.load('emg_scaler.pkl')
#    print("Model and Scaler loaded.")
#except FileNotFoundError:
#    print("Error: Model files not found. Run train_model.py first.")
#    exit()

# 2. Setup Signal Processing Parameters
fs = 200.0
nyq = 0.5 * fs
b_band, a_band = butter(4, [20.0/nyq, 99.0/nyq], btype='band')
w0 = 60.0 / nyq
b_notch, a_notch = iirnotch(w0, 30)

# Window configuration
window_size = 50 # 250ms at 200Hz
# We use a slightly larger buffer for filtering to avoid edge artifacts on small windows
filter_buffer_size = 200 
emg_buffer = deque(maxlen=filter_buffer_size)

# 3. Connect to LSL
print("Looking for an EMG stream...")
streams = resolve_byprop('type', 'EEG') # Adjust 'type' to your LSL stream configuration
inlet = StreamInlet(streams[0])
print("Connected to stream.")

# 4. Live Prediction Loop
print("Starting live prediction...")
while True:
    chunk, timestamps = inlet.pull_chunk()
    
    if chunk:
        # Assuming single channel EMG at index 0 of the chunk
        for sample in chunk:
            emg_buffer.append(sample[0]) 

        # Only process if we have enough data to filter and extract a window
        if len(emg_buffer) == filter_buffer_size:
            
            # Convert buffer to array
            raw_signal = np.array(emg_buffer)
            
            # Apply filters to the whole buffer
            filtered = lfilter(b_band, a_band, raw_signal)
            filtered = lfilter(b_notch, a_notch, filtered)
            
            # Extract only the most recent window_size (50 samples) for feature calculation
            win = filtered[-window_size:]
            
            # Calculate time-domain features
            rms = np.sqrt(np.mean(win**2))
            mav = np.mean(np.abs(win))
            zc = np.sum(np.diff(np.sign(win)) != 0)
            wl = np.sum(np.abs(np.diff(win)))
            
            features = np.array([[rms, mav, zc, wl]])
            
            # Scale features using the saved scaler
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = clf.predict(features_scaled)[0]
            probabilities = clf.predict_proba(features_scaled)[0]
            max_prob = np.max(probabilities)
            
            # Control Logic Mapping
            if max_prob < 0.75:
                # Confidence threshold to prevent jitter
                command = "STOP (Low Confidence)"
            elif prediction == 'rest':
                command = "STOP"
            elif prediction == 'flex':
                command = "FORWARD"
            elif prediction == 'extend':
                command = "REVERSE"
            else:
                command = "UNKNOWN"
                
            print(f"Command: {command} | Confidence: {max_prob:.2f} | Features: RMS={rms:.2f}")

    # Small sleep to prevent maxing out CPU
    time.sleep(0.01)