# Serial Bridge — Wheelchair Classifier → ESP32

Connects the live EMG classifier to an ESP32 over USB serial. The ESP32 changes its built-in LED blink pattern based on the predicted movement command. Motor control is **not included** — this is a safe signal-verification step first.

---

## A. What this does

```
live_predict.py            (original — untouched)
        |
live_predict_wrapper.py    (importable version of the same logic)
        ↓
serial_bridge.py           (maps labels to serial commands)
        ↓
run_live_mock_test.py      ← START HERE (no hardware needed at all)
     OR
run_live_to_serial.py      (use after mock test passes + ESP32 connected)
        ↓
arduino/wheelchair_serial_test/wheelchair_serial_test.ino
        ↓
Built-in LED on ESP32:
   "0" → off         (STOP / rest)
   "1" → slow blink  (FORWARD / flex)
   "2" → fast blink  (REVERSE / extend)
```

### Files added (originals never modified)

| File | Purpose |
|---|---|
| `live_predict_wrapper.py` | Importable duplicate of prediction logic — exposes `live_prediction_stream()` generator |
| `serial_bridge.py` | `ESP32SerialBridge` class — mock or real serial |
| `run_live_mock_test.py` | **Start here** — verifies everything on laptop only, zero hardware |
| `run_live_to_serial.py` | Live EMG → real ESP32 (use after mock test passes) |
| `arduino/wheelchair_serial_test/wheelchair_serial_test.ino` | ESP32 LED sketch |
| `README_serial_bridge.md` | This file |

`live_predict.py`, `LSL.py`, `multi_channel_predict.py`, and the `.pkl` model files are **not changed**.

---

## B. Setup

### Clone the repo

```bash
git clone https://github.com/aghosh376/WheelchairClassification.git
cd WheelchairClassification
```

### Install Python dependencies

```bash
python3 -m pip install pyserial
```

If the repo has a `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

Otherwise install the core stack manually:

```bash
python3 -m pip install numpy scipy scikit-learn joblib pylsl
```

> `pylsl` is only needed for live EMG prediction (`run_live_to_serial.py`).  
> The laptop mock test (`run_live_mock_test.py`) does **not** need `pylsl` or any hardware.

---

## C. Laptop-only test — no ESP32, no EMG sensor needed

> **Run this first.** It confirms the signal pipeline works before connecting anything.

```bash
python3 run_live_mock_test.py
```

### What it does

- Cycles through simulated predictions (`FORWARD`, `STOP`, `REVERSE`, etc.) — no sensor required.
- Passes each prediction through `ESP32SerialBridge(mock=True)`.
- Prints what **would** be sent to the ESP32.  Nothing is actually transmitted.

### Expected output

```
============================================================
  LAPTOP-ONLY MOCK TEST — no ESP32 or EMG hardware needed
============================================================

Using SIMULATED predictions.
Press Ctrl+C to stop early.
------------------------------------------------------------
[MockBridge] Mock mode active — no serial port needed.
[MockBridge] Would connect to port=None, baud=115200

[MockBridge] MOCK SEND TO ESP32: 1
Raw prediction : FORWARD                      (confidence=0.87)
Mapped command : 1

[MockBridge] MOCK SEND TO ESP32: 0
Raw prediction : STOP                         (confidence=0.92)
Mapped command : 0

[MockBridge] MOCK SEND TO ESP32: 2
Raw prediction : REVERSE                      (confidence=0.81)
Mapped command : 2
```

Press `Ctrl+C` to stop early, or let it run through the simulated sequence.

### What this confirms

- Label-to-command mapping is correct.
- The bridge code is wired up properly.
- The prediction loop structure works end to end.

---

## D. ESP32 setup (after mock test passes)

### 1. Upload the Arduino sketch

1. Open **Arduino IDE**.
2. **File → Open** → navigate to and open:
   ```
   arduino/wheelchair_serial_test/wheelchair_serial_test.ino
   ```
3. **Tools → Board** → select your ESP32 board (e.g. *ESP32 Dev Module*).
4. Connect the ESP32 over USB.
5. **Tools → Port** → select your ESP32's port (see step 3 below for how to find it).
6. Click **Upload** (the → arrow).
7. Wait for *Done uploading.*

### 2. Test the sketch manually with Serial Monitor

1. **Tools → Serial Monitor**.
2. Set baud to **115200** in the bottom-right dropdown.
3. Type `1`, press Enter → LED blinks slowly.
4. Type `2`, press Enter → LED blinks fast.
5. Type `0`, press Enter → LED turns off.
6. **Close Serial Monitor** before running Python. Both cannot share the port at the same time.

### 3. Find your ESP32's serial port (macOS)

```bash
ls /dev/cu.*
```

Plug the ESP32 in, run the command again, and look for what is **new**. Common names:

```
/dev/cu.usbserial-0001
/dev/cu.SLAB_USBtoUART
/dev/cu.wchusbserial10
/dev/cu.usbmodem14101
```

### 4. Set the port in the Python script

Open `run_live_to_serial.py` and change the `PORT` line near the top:

```python
PORT = "/dev/cu.usbserial-0001"   # <-- your port here
```

### 5. Run the live bridge

Make sure:
- EMG hardware is connected and LSL stream is running.
- Arduino Serial Monitor is **closed**.
- Correct port is set in `run_live_to_serial.py`.

```bash
python3 run_live_to_serial.py
```

The ESP32 LED will now respond to live EMG predictions in real time.

---

## E. Troubleshooting

### `No DFU capable USB device available`
The ESP32 is not in bootloader mode. Hold the **BOOT** button on the ESP32 while clicking Upload in Arduino IDE. Release once uploading starts.

### `Port not found` / `could not open port`
- Close Arduino Serial Monitor — it holds the port exclusively.
- Confirm the port name with `ls /dev/cu.*`.
- Try a different USB cable (some cables are power-only, no data).
- Check the port in `run_live_to_serial.py` matches exactly, including capitalisation.

### `Permission denied on /dev/cu.*`
```bash
sudo chmod 666 /dev/cu.usbserial-0001
```

### `No module named serial`
```bash
python3 -m pip install pyserial
```
Make sure you are running the same Python environment used to install packages.

### `Serial Monitor shows gibberish`
Baud rate mismatch. Set Serial Monitor to **115200** — this must match `Serial.begin(115200)` in the sketch and `BAUD = 115200` in the Python script.

### `LED not blinking`
- Confirm the sketch uploaded without errors.
- Open Serial Monitor, type `1`, check for `Command received: 1`.
- Some boards have the built-in LED on GPIO 0 or GPIO 13 instead of GPIO 2. Check your board's pinout and update `LED_PIN` in the `.ino` if needed.

### `live_predict_wrapper hangs at "Looking for an EMG stream..."`
The LSL stream is not running. Start your EMG data source first (OpenBCI, BrainFlow, a test LSL stream, etc.). The mock test does not need this.

### `Arduino Serial Monitor and Python both need the port`
This is expected — the serial port is exclusive. Always close Serial Monitor before running the Python bridge script.

---

## Future: Adding Wheelchair Motor Control

When motor control is ready, look for comments marked **`FUTURE MOTOR CONTROL`** in:
- `run_live_to_serial.py` — where to add motor calls in Python
- `arduino/wheelchair_serial_test/wheelchair_serial_test.ino` — where to add motor code on the ESP32

The command set will expand from 3 LED modes to full directional control as more EMG gesture classes (left, right, etc.) are trained and added to `COMMAND_MAP` in `serial_bridge.py`.
