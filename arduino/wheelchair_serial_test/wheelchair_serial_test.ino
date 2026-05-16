/*
  wheelchair_serial_test.ino
  ESP32 — LED blink test controlled over USB serial.

  Serial commands (single character, sent from Python):
    '0'  → LED off             (STOP / rest)
    '1'  → slow blink 500 ms   (FORWARD / flex)
    '2'  → fast blink 100 ms   (REVERSE / extend)

  Baud rate: 115200 — must match the Python bridge and Serial Monitor.

  Built-in LED: GPIO 2 on most ESP32 devkit boards.
  If your board uses a different pin, update LED_PIN below.

  Uses millis() throughout — loop() is never blocked by delay().

  No motor control is included.  When motor control is added, look for
  the "FUTURE MOTOR CONTROL" comment inside loop().
*/

// ── Configuration ────────────────────────────────────────────────────────
const int LED_PIN = 2;

const unsigned long SLOW_INTERVAL = 500;   // ms on/off for command '1'
const unsigned long FAST_INTERVAL = 100;   // ms on/off for command '2'
// ─────────────────────────────────────────────────────────────────────────

char         currentCommand  = '0';
unsigned long lastToggleTime = 0;
bool         ledState        = false;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("ESP32 wheelchair serial test ready.");
  Serial.println("Send: '0' = LED off  |  '1' = slow blink  |  '2' = fast blink");
}

void loop() {
  // 1. Read incoming serial command
  if (Serial.available() > 0) {
    char incoming = (char)Serial.read();

    // Ignore newline / carriage return so Python's "\n"-terminated writes work
    if (incoming == '\n' || incoming == '\r') {
      return;
    }

    if (incoming == '0' || incoming == '1' || incoming == '2') {
      currentCommand = incoming;
      Serial.print("Command received: ");
      Serial.println(currentCommand);

      // ── FUTURE MOTOR CONTROL ─────────────────────────────────────────
      // Map commands to wheelchair motors here when ready:
      //   '0' → stop motors
      //   '1' → drive forward
      //   '2' → reverse  (expand when left/right classes are added)
      // ─────────────────────────────────────────────────────────────────

    } else {
      Serial.print("Unknown command ignored: ");
      Serial.println(incoming);
    }
  }

  // 2. Non-blocking LED update
  unsigned long now = millis();

  if (currentCommand == '0') {
    digitalWrite(LED_PIN, LOW);
    ledState = false;

  } else {
    unsigned long interval = (currentCommand == '1') ? SLOW_INTERVAL : FAST_INTERVAL;

    if (now - lastToggleTime >= interval) {
      lastToggleTime = now;
      ledState = !ledState;
      digitalWrite(LED_PIN, ledState ? HIGH : LOW);
    }
  }
}
