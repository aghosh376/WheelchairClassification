"""
serial_bridge.py — reusable ESP32 serial bridge.

Works in two modes:
  mock=True  — prints what WOULD be sent; no hardware needed at all.
  mock=False — opens a real serial port and sends commands to the ESP32.

Command mapping (matches commands produced by live_predict_wrapper.py):
  "FORWARD"              → "1"   (drive forward)
  "STOP"                 → "0"   (stop / hold)
  "COAST"                → "2"   (no active drive, free-roll)
  "STOP (Low Confidence)"→ "0"   (safety default)
  "UNKNOWN"              → "0"   (safety default)

Byte values must match the switch/case in wheelchair_serial_test.ino.
Add new direction labels to COMMAND_MAP when left/right classes are trained.
"""

import time

# ── Label → serial command mapping ───────────────────────────────────────────
COMMAND_MAP = {
    # live_predict_wrapper.py outputs
    "FORWARD"              : "1",
    "STOP"                 : "0",
    "COAST"                : "2",
    "STOP (Low Confidence)": "0",
    "UNKNOWN"              : "0",

    # Raw motion labels (in case called directly with clf output)
    "flex"  : "1",
    "extend": "0",
    "rest"  : "2",

    # Generic aliases for manual testing
    "stop"   : "0",
    "off"    : "0",
    "neutral": "0",
    "forward": "1",
    "coast"  : "2",
    "reverse": "2",
    "left"   : "1",
    "right"  : "2",

    # Raw byte pass-through
    "0": "0",
    "1": "1",
    "2": "2",
}

VALID_COMMANDS = {"0", "1", "2"}


class ESP32SerialBridge:
    """
    Translates classifier labels into serial commands for the ESP32.

    Parameters
    ----------
    port : str | None
        Serial port, e.g. '/dev/cu.usbserial-0001'. Ignored in mock mode.
    baud : int
        Baud rate — must match Serial.begin() in the Arduino sketch (115200).
    startup_delay : float
        Seconds to wait after opening port so the ESP32 can finish booting.
    mock : bool
        If True, no hardware is needed; commands are printed instead of sent.
    """

    def __init__(self, port=None, baud=115200, startup_delay=2, mock=False):
        self.port          = port
        self.baud          = baud
        self.startup_delay = startup_delay
        self.mock          = mock
        self._serial       = None

    def connect(self):
        """Open the serial connection (or confirm mock mode is ready)."""
        if self.mock:
            print("[MockBridge] Mock mode active — no serial port needed.")
            print(f"[MockBridge] Would connect to port={self.port}, baud={self.baud}")
            return

        try:
            import serial
        except ImportError:
            raise ImportError(
                "pyserial is not installed.\n"
                "Run:  pip install pyserial"
            )

        if not self.port:
            raise ValueError(
                "Serial port not set.\n"
                "Edit PORT in run_live_to_serial.py.\n"
                "Find available ports with:  ls /dev/cu.*   (macOS)\n"
                "                            ls /dev/tty*   (Linux)\n"
                "                            mode           (Windows)"
            )

        import serial as _serial_mod
        self._serial = _serial_mod.Serial(self.port, self.baud, timeout=1)
        print(f"[Bridge] Connected to {self.port} at {self.baud} baud.")
        print(f"[Bridge] Waiting {self.startup_delay}s for ESP32 to boot...")
        time.sleep(self.startup_delay)
        print("[Bridge] Ready.")

    def send_command(self, command_or_label: str) -> str:
        """
        Send a command to the ESP32.

        Accepts a high-level label ("FORWARD", "STOP", "COAST") or a raw
        command string ("0", "1", "2"). Unknown labels default safely to "0".

        Returns the raw command string that was sent (or would be sent).
        """
        raw_cmd = COMMAND_MAP.get(str(command_or_label))

        if raw_cmd is None:
            print(
                f"[Bridge] WARNING: unknown label '{command_or_label}' "
                f"— defaulting to STOP (0)."
            )
            raw_cmd = "0"

        if raw_cmd not in VALID_COMMANDS:
            print(
                f"[Bridge] WARNING: invalid command '{raw_cmd}' "
                f"— defaulting to STOP (0)."
            )
            raw_cmd = "0"

        if self.mock:
            print(f"[MockBridge] MOCK SEND → ESP32: '{raw_cmd}'  "
                  f"(from label: '{command_or_label}')")
        else:
            if self._serial is None or not self._serial.is_open:
                raise RuntimeError(
                    "Bridge not connected. Call connect() first."
                )
            self._serial.write((raw_cmd + "\n").encode("utf-8"))

        return raw_cmd

    def close(self):
        """Close the serial port if open."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            print(f"[Bridge] Port {self.port} closed.")
        elif self.mock:
            print("[MockBridge] Mock session ended.")
