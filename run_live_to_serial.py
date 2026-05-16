"""
run_live_to_serial.py — live EMG classifier → serial → ESP32.

Prerequisites before running:
  1. EMG hardware connected and LSL stream running.
  2. ESP32 flashed with arduino/wheelchair_serial_test/wheelchair_serial_test.ino.
  3. Arduino Serial Monitor CLOSED (only one app can hold the port at a time).
  4. Correct port set in the PORT variable below.

How to find your port on macOS:
    ls /dev/cu.*

  Plug the ESP32 in, run that command again, and note what is new.
  Common names:
    /dev/cu.usbserial-0001
    /dev/cu.SLAB_USBtoUART
    /dev/cu.wchusbserial10
    /dev/cu.usbmodem14101

Then run:
    python3 run_live_to_serial.py
"""

# ── Configure these before running ─────────────────────────────────────────
PORT          = "/dev/cu.usbserial-0001"  # <-- change to your ESP32's port
BAUD          = 115200                    # must match Serial.begin(115200) in the sketch
STARTUP_DELAY = 2                         # seconds to wait for ESP32 to boot after connect
# ───────────────────────────────────────────────────────────────────────────

from serial_bridge import ESP32SerialBridge
from live_predict_wrapper import load_model, connect_lsl_inlet, live_prediction_stream


def main():
    print("=" * 60)
    print("  LIVE EMG → ESP32 SERIAL BRIDGE")
    print("=" * 60)
    print(f"Port: {PORT}  |  Baud: {BAUD}")
    print()
    print("Make sure Arduino Serial Monitor is CLOSED before continuing.")
    print("Press Ctrl+C to stop.")
    print("-" * 60)

    clf, scaler = load_model()
    inlet       = connect_lsl_inlet()

    bridge = ESP32SerialBridge(port=PORT, baud=BAUD, startup_delay=STARTUP_DELAY, mock=False)
    bridge.connect()

    print("\nLive prediction loop running...\n")

    try:
        for result in live_prediction_stream(inlet, clf, scaler):
            raw_prediction = result["command"]
            confidence     = result["confidence"]
            rms            = result["rms"]

            serial_command = bridge.send_command(raw_prediction)

            print(
                f"Prediction: {raw_prediction:<28} "
                f"cmd={serial_command}  conf={confidence:.2f}  RMS={rms:.4f}"
            )

            # ── FUTURE MOTOR CONTROL goes here ──────────────────────────
            # When wheelchair drive is added, map serial_command to motors:
            #   "0" → stop motors
            #   "1" → drive forward
            #   "2" → reverse  (expand to left/right once trained)
            # ─────────────────────────────────────────────────────────────

    except KeyboardInterrupt:
        print("\n[Bridge] Stopped by user.")
    finally:
        bridge.close()
        print("[Bridge] Done.")


if __name__ == "__main__":
    main()
