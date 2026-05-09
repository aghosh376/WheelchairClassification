# TNT Wheelchair EMG Controller

A real-time EMG-based wheelchair control system using a pre-trained SVM classifier.
Four EMG electrode patches (two per arm) stream signals through an OpenBCI Ganglion,
which are classified live and sent as movement commands to an Arduino wheelchair controller.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Requirements](#software-requirements)
5. [Wiring and Electrode Placement](#wiring-and-electrode-placement)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Testing — Step by Step](#testing--step-by-step)
9. [Running the System](#running-the-system)
10. [Troubleshooting](#troubleshooting)
11. [Model Details](#model-details)
12. [Command Reference](#command-reference)

---

## How It Works

```
EMG Electrodes (x4)
      │
      ▼
OpenBCI Ganglion Board
      │  (4-channel EEG/EMG stream via BrainFlow)
      ▼
Lab Streaming Layer (LSL)   ← obci_eeg1 stream, 200 Hz
      │
      ▼
multi_channel_predict.py
  ├─ Bandpass filter: 20–99 Hz (removes motion artifact and noise)
  ├─ Notch filter: 60 Hz (removes US power line interference)
  ├─ Sliding window: 100 samples (0.5s at 200 Hz)
  ├─ Feature extraction per channel: [RMS, MAV, ZC, WL, MNF, SSC]
  ├─ StandardScaler (emg_scaler.pkl) — normalizes features to training distribution
  ├─ SVM classifier (emg_svm_model.pkl) — outputs [extend, flex, rest] probabilities
  ├─ Arbitration — picks highest-confidence command across all 4 channels
  └─ Smoothing — majority vote over last 10 predictions
      │
      ▼
Arduino (Serial, 9600 baud)
      │
      ▼
Wheelchair Motor Controller
```

The model runs **independently on each channel** — it is a single-channel classifier
called 4 times per window. This means it works even if some electrodes have poor contact,
and does not need to be retrained if electrode positions shift slightly between sessions.

---

## Project Structure

```
WheelchairClassification/
├── emg_svm_model.pkl        # Trained SVM classifier (do not modify)
├── emg_scaler.pkl           # StandardScaler fit to training data (do not modify)
├── LSL.py                   # Step 1: Diagnostic — verify stream and sampling rate
├── live_predict.py          # Step 2: Single-channel sanity check
├── multi_channel_predict.py # Step 3: Full 4-channel production controller
└── README.md                # This file
```

---

## Hardware Requirements

| Component | Details |
|---|---|
| OpenBCI Ganglion | 4-channel EEG/EMG board |
| EMG Electrode Patches | 4 total — 2 per arm, one upper and one lower |
| Bluetooth or USB dongle | For Ganglion connection to PC |
| Arduino (Uno or Mega) | Receives serial commands from PC |
| Wheelchair motor controller | Wired to Arduino outputs |
| PC / Laptop | Runs Python scripts; macOS or Windows |

---

## Software Requirements

```
Python 3.9+
pylsl
numpy
scipy
scikit-learn == 1.6.1    ← pin this version to match the .pkl files
joblib
pyserial
brainflow                ← for OpenBCI Ganglion streaming
```

> **Important:** The `.pkl` files were saved with scikit-learn `1.6.1`.
> Using a different version may cause warnings or incorrect results.
> Always use a virtual environment (see Installation).

---

## Wiring and Electrode Placement

### Ganglion Pin → Channel Mapping

```
Ganglion Pin 1 (sample[0]) → Ch0 → Right arm, UPPER patch  (bicep area)
Ganglion Pin 2 (sample[1]) → Ch1 → Right arm, LOWER patch  (forearm area)
Ganglion Pin 3 (sample[2]) → Ch2 → Left arm,  UPPER patch  (bicep area)
Ganglion Pin 4 (sample[3]) → Ch3 → Left arm,  LOWER patch  (forearm area)
```

If you wire your patches differently, update `CHANNEL_ROLES` at the top of
`multi_channel_predict.py` to match. The arbitration logic does not depend on
a specific layout — it finds the best signal across all 4 regardless.

### Electrode Placement Tips

- Place electrodes **along the muscle belly**, not over tendons or bone.
- The reference electrode (SRB pin on Ganglion) goes on a bony landmark —
  elbow tip or wrist bone works well.
- Clean the skin with an alcohol wipe before attaching.
- If signal is noisy, press firmly on the patch for 10 seconds to improve contact.

### Arduino Serial Commands

| Command sent | Meaning | Expected wheelchair action |
|---|---|---|
| `F\n` | FORWARD (flex detected) | Drive forward |
| `S\n` | STOP (extend detected) | Brake / stop |
| `C\n` | COAST (no clear signal) | No change / coast |

---

## Installation

```bash
# 1. Clone or copy the project folder to your machine
cd WheelchairClassification

# 2. Create a virtual environment
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies — pin scikit-learn to match the .pkl files
pip install pylsl numpy scipy joblib pyserial brainflow scikit-learn==1.6.1
```

---

## Configuration

All tunable parameters are at the top of `multi_channel_predict.py`:

```python
CONFIDENCE_THRESHOLD = 0.60   # raise to 0.70–0.75 if getting false triggers
PREDICTION_WINDOW    = 10     # raise to 15–20 for smoother but slower response
WHEELCHAIR_PORT      = None   # None = auto-detect; or set 'COM3' / '/dev/ttyUSB0'
WHEELCHAIR_BAUDRATE  = 9600   # must match your Arduino sketch
```

For `live_predict.py` (single-channel debug), the threshold is separately set to `0.75`
because single-channel classification is less reliable than 4-channel arbitration.

---

## Testing — Step by Step

Run these steps **in order** every session, especially before using the wheelchair.
Do not skip to `multi_channel_predict.py` without completing Steps 1 and 2.

---

### Step 1 — Verify the LSL Stream (`LSL.py`)

**Purpose:** Confirm the Ganglion is streaming and all 4 channels are producing data.

```bash
python LSL.py
```

**What to look for:**

```
[OK] Found: 'obci_eeg1' | Type: EEG | Channels: 4 | Rate: 200.0 Hz
Testing for 10s — reading all channels...
  [chunk   1]  10 samples | First: Ch0: -0.000012 | Ch1: 0.000034 | Ch2: -0.000008 | Ch3: 0.000021
  ...
Avg sample rate : 199.8 Hz
[OK] Sampling rate looks good.
```

**Failure modes and fixes:**

| What you see | Cause | Fix |
|---|---|---|
| `[ERROR] No streams found` | Ganglion not streaming | Open OpenBCI GUI, start session, confirm BrainFlow is running |
| `Channels: 1` instead of `4` | Wrong stream grabbed | Check OpenBCI GUI channel settings; all 4 should be enabled |
| `Avg sample rate: 98 Hz` | Bluetooth congestion | Move closer to dongle, close other Bluetooth devices |
| All Ch values are exactly `0.000000` | Electrode not connected | Check Ganglion lead wires; reseat connectors |
| One channel always `0.000000` | That Ganglion pin disconnected | Reseat that specific lead wire |

**Do not proceed to Step 2 until:**
- Stream name is `obci_eeg1`
- Channel count is `4`
- Sampling rate is between 195–205 Hz
- All 4 channel values are non-zero and vary between chunks

---

### Step 2 — Single-Channel Sanity Check (`live_predict.py`)

**Purpose:** Verify the model is producing sensible predictions on one real electrode
before involving all 4. This isolates model/scaler issues from multi-channel issues.

```bash
python live_predict.py
```

This reads **Ch0 only** (right arm upper patch).

**What to look for — at rest (arm relaxed):**

```
COAST  (rest)                | conf=0.82 | flex=0.08 ext=0.11 rest=0.82 | RMS=12.3µV
COAST  (rest)                | conf=0.79 | flex=0.10 ext=0.09 rest=0.79 | RMS=14.1µV
```

- Rest confidence should be above 0.75 when your arm is completely still.
- RMS at rest should be low — typically 5–30 µV.

**What to look for — during flex (curl arm toward shoulder):**

```
FORWARD                      | conf=0.88 | flex=0.88 ext=0.06 rest=0.06 | RMS=312.4µV
FORWARD                      | conf=0.91 | flex=0.91 ext=0.04 rest=0.05 | RMS=445.2µV
```

- Flex confidence should clearly exceed 0.75 during the motion.
- RMS during movement should be 5–50x higher than at rest.

**What to look for — during extend (push arm straight out):**

```
COAST  (low confidence)      | conf=0.55 | flex=0.22 ext=0.55 rest=0.23 | RMS=289.1µV
```

This is expected on Ch0 — it is the upper patch which is not optimized for extend.
The extend signal will be detected properly by Ch1 in Step 3.

**Failure modes and fixes:**

| What you see | Cause | Fix |
|---|---|---|
| `COAST (low confidence)` even during strong flex | Electrode contact poor | Re-clean skin, press patch firmly, re-run |
| RMS stays below 20 µV during movement | Signal not reaching model in µV range | Confirm `SCALE_FACTOR = 1_000_000` in script |
| `rest` confidence < 0.50 even when arm still | Electrical noise / bad reference | Move away from power supplies; check reference electrode |
| `ValueError: X has 6 features but scaler expects N` | Wrong pkl file loaded | Confirm `emg_svm_model.pkl` and `emg_scaler.pkl` are in same folder as script |
| All predictions always `extend` | Feature out-of-distribution | Check `WINDOW_SIZE = 100` matches — do not change to 50 |

**Do not proceed to Step 3 until:**
- Rest produces `COAST (rest)` with confidence > 0.75 reliably
- Strong flex produces `FORWARD` with confidence > 0.75

---

### Step 3 — 4-Channel Dry Run (No Wheelchair, `multi_channel_predict.py`)

**Purpose:** Verify all 4 channels are classifying correctly before connecting
the wheelchair. Run with the Arduino unplugged — the script handles this gracefully
and runs in simulation mode.

```bash
python multi_channel_predict.py
```

You will see:
```
[WARNING] No serial ports found.
[INFO] Running in SIMULATION mode — no serial output.
```

That is expected. Keep the wheelchair disconnected for now.

**What the output looks like:**

```
======================================================================
[WINDOW 7/10]
  Ch0 (R upper): flex= 82.3%  ext= 9.1%  rest= 8.6% ◄ winner
  Ch1 (R lower): flex= 44.1%  ext=38.2%  rest=17.7%
  Ch2 (L upper): flex= 71.2%  ext=12.3%  rest=16.5%
  Ch3 (L lower): flex= 28.4%  ext=55.1%  rest=16.5%
  Raw:      [F] FORWARD (conf=82.3%)
  SMOOTHED: [F] FORWARD (avg=79.4%) votes F=8 S=1 C=1
```

**Test sequence to run:**

1. **Arm fully at rest** — all channels should show `rest > 60%`, command should settle on `COAST`.
2. **Flex right arm** — Ch0 and/or Ch2 should show `flex > 60%`, command should become `FORWARD`.
3. **Extend right arm** — Ch1 and/or Ch3 should show `ext > 60%`, command should become `STOP`.
4. **Alternate flex → rest → extend** repeatedly — watch the vote counts reset cleanly each time.
5. **Cover one electrode with your hand (disconnect it mentally)** — the remaining 3 channels should still produce the correct command.

**What to check per channel:**

| Channel | Motion it should respond to | Confident class |
|---|---|---|
| Ch0 (R upper) | Flex (bicep curl) | `flex > 60%` during curl |
| Ch1 (R lower) | Extend (forearm push) | `ext > 60%` during extension |
| Ch2 (L upper) | Flex | `flex > 60%` during curl |
| Ch3 (L lower) | Extend | `ext > 60%` during extension |

**If a channel shows wrong dominant class:**
- The electrode may be on the wrong muscle — swap Ch0/Ch1 or Ch2/Ch3 physically, or update `CHANNEL_ROLES` in the script.
- It may be picking up cross-talk from the adjacent patch — spread patches further apart.

**Do not proceed to Step 4 until:**
- Each channel responds to its intended motion above threshold
- SMOOTHED command matches intended motion consistently
- COAST appears reliably when arm is at rest

---

### Step 4 — Arduino Serial Test (No Wheelchair Movement)

**Purpose:** Verify serial communication works before the wheelchair can move.
Power the Arduino but do not connect it to the motor controller yet.

Upload this minimal sketch to the Arduino first:

```cpp
void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'F') Serial.println("GOT: FORWARD");
    if (cmd == 'S') Serial.println("GOT: STOP");
    if (cmd == 'C') Serial.println("GOT: COAST");
  }
}
```

Then run `multi_channel_predict.py` with the Arduino connected. In the console you should see:
```
[OK] Wheelchair connected on /dev/cu.usbmodem14201
...
>>> SENT 'F' to wheelchair
```

Open Arduino Serial Monitor (9600 baud) in parallel — you should see:
```
GOT: FORWARD
GOT: COAST
GOT: STOP
```

**Only proceed to Step 5 once Arduino is echoing the correct commands.**

---

### Step 5 — Full System Test (Wheelchair on Jack Stand)

**Purpose:** First live run with the wheelchair motor controller connected,
but with the wheelchair raised off the ground so wheels spin freely.

1. Raise the wheelchair on a jack stand or flip it so wheels are off the ground.
2. Connect Arduino to motor controller.
3. Run `multi_channel_predict.py`.
4. Flex → verify wheels spin forward.
5. Extend → verify wheels stop or brake.
6. Rest → verify no movement.

Test for at least **2 minutes** of mixed motions before lowering the wheelchair.

---

## Running the System

Once all 5 test steps pass:

```bash
# Activate your virtual environment first
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Run the full controller
python multi_channel_predict.py
```

Press **Ctrl+C** to stop. The serial port is closed cleanly on exit.

---

## Troubleshooting

### "No LSL stream found"
- Open OpenBCI GUI and start a session before running any script.
- Confirm the stream name is `obci_eeg1`. If different, update `target_name` in `connect_lsl()`.

### Commands are jittery / switching rapidly
- Raise `CONFIDENCE_THRESHOLD` from `0.60` to `0.70`.
- Raise `PREDICTION_WINDOW` from `10` to `15`.
- Check electrode contact — poor contact produces noise that confuses the classifier.

### Always predicts "extend" or always "flex"
- The signal is likely not in microvolts. Confirm `SCALE_FACTOR = 1_000_000`.
- The wrong `win_size` was used somewhere — it must be `100` everywhere.

### Serial port not found on Windows
- Set `WHEELCHAIR_PORT = 'COM3'` (or whatever port Device Manager shows).

### Serial port not found on Mac
- Set `WHEELCHAIR_PORT = '/dev/cu.usbmodem14201'` (run `ls /dev/cu.*` to find yours).

### scikit-learn version warning
```
InconsistentVersionWarning: Trying to unpickle estimator SVC from version 1.6.1
```
Fix: `pip install scikit-learn==1.6.1`

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | SVM — `sklearn.svm.SVC` |
| Kernel | RBF |
| C | 10 |
| Gamma | `scale` (= 1 / n_features / X.var() ≈ 0.167) |
| Class weight | `balanced` (compensates for fewer rest samples) |
| Classes | `extend`, `flex`, `rest` |
| Features | 6 per channel: RMS, MAV, ZC, WL, MNF, SSC |
| Window size | 100 samples (0.5s at 200 Hz) |
| Training samples | ~61,845 windows |
| Scaler | `StandardScaler` — zero mean, unit variance |

The model was trained on **single-channel windows**. It does not know about
multi-channel context — the 4-channel architecture is handled entirely in
`multi_channel_predict.py` by running the model once per channel and combining results.

---

## Command Reference

| Script | Purpose | When to use |
|---|---|---|
| `LSL.py` | Stream diagnostic | Every session, run first |
| `live_predict.py` | Single-channel debug | Verifying one electrode; model sanity check |
| `multi_channel_predict.py` | Full 4-channel controller | Production use |