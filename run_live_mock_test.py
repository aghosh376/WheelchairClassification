"""
run_live_mock_test.py — laptop-only test, zero hardware required.

Confirms the full signal → bridge pipeline works before connecting any hardware.

Does NOT require:
  - An ESP32 / serial port
  - An EMG sensor or LSL stream
  - pylsl installed

What it tests:
  1. The label-to-command mapping in serial_bridge.py is correct.
  2. The mock serial output is formatted the way the ESP32 expects.
  3. The prediction loop structure mirrors run_live_to_serial.py exactly.

Once this passes cleanly, connect the ESP32 and run run_live_to_serial.py.
"""

import time
import random
from serial_bridge import ESP32SerialBridge

# ---------------------------------------------------------------------------
# Simulated prediction stream
# Produces the same output format as live_predict_wrapper.live_prediction_stream()
# but generates fake data — no hardware needed at all.
# ---------------------------------------------------------------------------

# These are the exact command strings live_predict.py can output.
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


def simulated_prediction_stream(delay=1.0):
    """
    Generator that yields fake prediction dicts matching live_predict_wrapper format.

    Parameters
    ----------
    delay : float
        Seconds between predictions (default 1 s, easy to read in the terminal).
    """
    for i, command in enumerate(_SIMULATED_SEQUENCE * 3):  # repeat 3 times then stop
        yield {
            "command":    command,
            "raw_label":  command.lower().replace(" ", "_"),
            "confidence": round(random.uniform(0.70, 0.99), 2),
            "rms":        round(random.uniform(0.01, 0.40), 4),
        }
        time.sleep(delay)


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  LAPTOP-ONLY MOCK TEST — no ESP32 or EMG hardware needed")
    print("=" * 60)
    print()
    print("Using SIMULATED predictions.")
    print("Press Ctrl+C to stop early.")
    print("-" * 60)

    bridge = ESP32SerialBridge(mock=True)
    bridge.connect()
    print()

    try:
        for result in simulated_prediction_stream(delay=1.0):
            raw_prediction = result["command"]
            confidence     = result["confidence"]

            serial_command = bridge.send_command(raw_prediction)

            print(f"Raw prediction : {raw_prediction:<28} (confidence={confidence:.2f})")
            print(f"Mapped command : {serial_command}")
            print()

    except KeyboardInterrupt:
        print("\n[Test] Stopped by user.")
    finally:
        bridge.close()
        print("[Test] Done.")
        print()
        print("If all mappings look correct, you are ready to connect the ESP32")
        print("and run:  python3 run_live_to_serial.py")


if __name__ == "__main__":
    main()
