"""
run_live_mock_test.py — test with real EMG, no ESP32 needed.

The bridge runs in mock mode: predictions from your real EMG arm movements
are printed to the screen exactly as they would be sent to the ESP32 —
but nothing is actually transmitted, so no Arduino is required.

Two modes (set USE_REAL_EMG below):

  USE_REAL_EMG = True   ← use this when your EMG sensor + LSL stream are running.
                          Real arm movements → real predictions → printed mock commands.

  USE_REAL_EMG = False  ← use this if you have NO hardware at all (pure software check).
                          Fake predictions cycle automatically, no sensor needed.
"""

import time
import random
from serial_bridge import ESP32SerialBridge

# ── Set this to True if your EMG sensor and LSL stream are running ──────────
USE_REAL_EMG = True
# ────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Simulated stream — used only when USE_REAL_EMG = False
# ---------------------------------------------------------------------------
_SIMULATED_SEQUENCE = [
    "FORWARD",
    "STOP",
    "REVERSE",
    "STOP (Low Confidence)",
    "UNKNOWN",
    "FORWARD",
    "FORWARD",
    "STOP",
    "REVERSE",
]

def _simulated_prediction_stream(delay=1.0):
    for command in _SIMULATED_SEQUENCE * 3:
        yield {
            "command":    command,
            "raw_label":  command.lower().replace(" ", "_"),
            "confidence": round(random.uniform(0.70, 0.99), 2),
            "rms":        round(random.uniform(0.01, 0.40), 4),
        }
        time.sleep(delay)


# ---------------------------------------------------------------------------
def main():
    bridge = ESP32SerialBridge(mock=True)

    if USE_REAL_EMG:
        print("=" * 60)
        print("  LIVE EMG TEST — no ESP32 needed")
        print("=" * 60)
        print()
        print("Your real arm movements will be classified and the")
        print("commands that WOULD go to the ESP32 are printed here.")
        print()
        print("Press Ctrl+C to stop.")
        print("-" * 60)

        # Import here so the script still runs (simulated mode) if pylsl is missing
        try:
            from live_predict_wrapper import load_model, connect_lsl_inlet, live_prediction_stream
        except ImportError as e:
            print(f"ERROR: Could not import live_predict_wrapper: {e}")
            print("Make sure pylsl and all model dependencies are installed.")
            return

        clf, scaler = load_model()
        inlet       = connect_lsl_inlet()
        stream      = live_prediction_stream(inlet, clf, scaler)

    else:
        print("=" * 60)
        print("  SIMULATED TEST — no hardware needed at all")
        print("=" * 60)
        print()
        print("Using fake predictions (not real EMG).")
        print("Set USE_REAL_EMG = True at the top of this file to use your sensor.")
        print()
        print("Press Ctrl+C to stop early.")
        print("-" * 60)

        stream = _simulated_prediction_stream(delay=1.0)

    bridge.connect()
    print()

    try:
        for result in stream:
            raw_prediction = result["command"]
            confidence     = result["confidence"]
            rms            = result["rms"]

            serial_command = bridge.send_command(raw_prediction)

            print(f"  Arm signal     : {raw_prediction:<28} (confidence={confidence:.2f}, RMS={rms:.4f})")
            print(f"  → ESP32 would receive: '{serial_command}'")
            print()

    except KeyboardInterrupt:
        print("\n[Test] Stopped.")
    finally:
        bridge.close()
        print("[Test] Done.")
        print()
        print("When the ESP32 is connected, run:  python3 run_live_to_serial.py")


if __name__ == "__main__":
    main()
