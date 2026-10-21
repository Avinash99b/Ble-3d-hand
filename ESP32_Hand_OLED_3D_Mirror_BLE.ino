/*
  ESP32-WROOM-32 / WROOM-DA  -  3D HAND MIRROR
  =====================================================================
  Camera device (hand_controller.html, MediaPipe hand tracking)
        |  BLE GATT writes in 20-byte chunks, 21 landmarks x (x,y,z) int16
        v
  ESP32  --> renders the hand in 3D on a 128x64 SSD1306 OLED (max FPS)
        --> relays the same frames to every other WebSocket client
  Compatible Chrome/Edge browser sends landmarks directly over BLE; no Wi-Fi on ESP32.

  Joystick (view control on the OLED, also synced to the web viewer):
    X axis ........ rotate left/right (yaw)
    Y axis ........ ORBIT mode: tilt (pitch)   |  ZOOM mode: zoom in/out
    Short press ... toggle ORBIT <-> ZOOM
    Long press .... reset view + re-centre the stick (let go of the stick first!)

  Wiring
    OLED  SDA -> GPIO21, SCL -> GPIO22, VCC -> 3V3, GND -> GND
    Joystick X -> GPIO32 (ADC1_CH4), Y -> GPIO35 (ADC1_CH7, input-only)
    Joystick SW -> GPIO14 (to GND when pressed, internal pull-up used)
    Joystick VCC -> 3V3 (NOT 5V: ADC pins must never see more than 3.3 V)

  Libraries (Library Manager)
    - Adafruit SSD1306  (+ Adafruit GFX, Adafruit BusIO)
    - WebSockets by Markus Sattler (arduinoWebSockets)
  Libraries: NimBLE-Arduino, Adafruit SSD1306 + GFX + BusIO.
  Board: ESP32 Dev Module / WROOM-DA, recent arduino-esp32 core.

  OLED is assumed to be SSD1306 128x64 @ 0x3C. For a 1.3" SH1106 module you need a
  different driver (e.g. U8g2 with the SH1106 constructor).
*/

#include <NimBLEDevice.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

// =====================================================================
//  USER SETTINGS
// =====================================================================
static const char* BLE_DEVICE_NAME = "ESP32-Hand-Mirror";
static const char* BLE_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
static const char* BLE_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
static constexpr uint8_t BLE_PACKET_MAGIC = 0xD7;
static constexpr uint8_t BLE_CHUNK_BYTES = 16;
static constexpr uint8_t BLE_CHUNK_COUNT = 8;
static constexpr size_t BLE_FRAME_BYTES = 128;

// ---- pins
#define PIN_OLED_SDA   21
#define PIN_OLED_SCL   22
#define PIN_JOY_X      32
#define PIN_JOY_Y      35
#define PIN_JOY_SW     14

// ---- OLED / I2C
#define OLED_W         128
#define OLED_H         64
#define OLED_ADDR      0x3C
#define I2C_HZ          400000UL   // conservative, tolerant of ordinary jumper-wire OLED wiring
#define OLED_FAST_OSC  0           // use the SSD1306 default oscillator for power/signal-margin debugging.

// ---- joystick conditioning
#define JOY_SAMPLES         8      // ADC reads per update; lower load and shorter calibration
#define JOY_TRIM            2      // discard N lowest + N highest of those reads
#define JOY_INIT_SPAN_MV  900.0f   // assumed half-travel in mV, auto-expands when exceeded
#define JOY_MIN_SPAN_MV   300.0f
#define JOY_NOISE_MV       12.0f   // changes below this are treated as noise (heavy smoothing)
#define JOY_FAST_MV       150.0f   // changes above this pass almost unfiltered
#define JOY_ALPHA_MIN      0.03f
#define JOY_ALPHA_MAX      0.60f
#define JOY_DEADZONE       0.10f   // fraction of full travel
#define JOY_EXPO           0.55f   // 0 = linear, 1 = fully quadratic (fine control near centre)
#define JOY_INVERT_X       false
#define JOY_INVERT_Y       false

// ---- view / rendering
#define YAW_RATE           2.6f    // rad/s at full deflection
#define PITCH_RATE         2.0f
#define ZOOM_RATE          1.3f    // e-folds/s
#define PITCH_LIMIT        1.45f
#define ZOOM_MIN           0.35f
#define ZOOM_MAX           2.60f
#define HAND_SMOOTH_TAU_MS 20.0f   // smoothing between camera frames (render runs faster than camera)
#define CAM_DIST           4.0f    // perspective distance (hand units)
#define HAND_PX_PER_UNIT   27.0f   // 1 unit ~ 12 cm
#define HUD_H              8

#define DEBOUNCE_MS        25
#define LONG_PRESS_MS      800
#define STALE_MS           1500    // no frames for this long -> "waiting"
#define FRAME_MAGIC        0xA5

// =====================================================================
//  Globals
// =====================================================================
Adafruit_SSD1306 display(OLED_W, OLED_H, &Wire, -1, I2C_HZ, I2C_HZ);
static NimBLEServer* g_bleServer = nullptr;
static NimBLECharacteristic* g_bleRx = nullptr;
static volatile bool g_bleConnected = false;
static uint8_t g_bleAssembly[BLE_FRAME_BYTES];
static uint8_t g_bleAssemblySeq = 0;
static uint8_t g_bleAssemblyMask = 0;
static bool g_bleAssemblyStarted = false;

// ---- data shared between network task (core 0) and render loop (core 1)
static portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
static int16_t  g_raw[63];
static volatile bool     g_present = false;
static volatile uint32_t g_seq = 0;
static volatile uint32_t g_lastRxMs = 0;
static volatile float    g_yaw = 0, g_pitch = 0, g_zoom = 1;

// ---- render-side state
static float g_cur[63], g_tgt[63];
static uint32_t g_lastSeq = 0, g_lastUs = 0;
enum Mode : uint8_t { MODE_ORBIT, MODE_ZOOM };
static Mode g_mode = MODE_ORBIT;
static uint32_t g_flashUntil = 0;
static int g_fps = 0;

// hand skeleton (MediaPipe 21-landmark topology)
static const uint8_t BONES[21][2] = {
  {0,1},{1,2},{2,3},{3,4},
  {0,5},{5,6},{6,7},{7,8},
  {5,9},{9,10},{10,11},{11,12},
  {9,13},{13,14},{14,15},{15,16},
  {13,17},{0,17},{17,18},{18,19},{19,20}
};

// =====================================================================
//  Joystick with noise rejection
//   1) 16x oversample, trimmed mean (kills spikes + WiFi-induced ADC noise)
//   2) adaptive EMA (heavy smoothing when still, fast when moving)
//   3) per-side normalisation using calibrated centre + learned travel
//   4) dead-zone (re-scaled so output starts at 0) + expo curve
// =====================================================================
struct JoyAxis {
  uint8_t pin = 0;
  bool invert = false;
  float center = 1650, lo = 0, hi = 0, filt = 1650;

  float sampleMv() const {
    uint16_t s[JOY_SAMPLES];
    for (uint8_t i = 0; i < JOY_SAMPLES; i++) s[i] = analogReadMilliVolts(pin);
    for (uint8_t i = 1; i < JOY_SAMPLES; i++) {            // insertion sort
      uint16_t v = s[i]; int8_t j = i - 1;
      while (j >= 0 && s[j] > v) { s[j + 1] = s[j]; j--; }
      s[j + 1] = v;
    }
    uint32_t sum = 0; uint8_t n = 0;
    for (uint8_t i = JOY_TRIM; i < JOY_SAMPLES - JOY_TRIM; i++) { sum += s[i]; n++; }
    return (float)sum / n;
  }

  void recenter() {
    // 12 trimmed samples are enough for neutral calibration without a long boot pause.
    float acc = 0;
    for (uint8_t i = 0; i < 12; i++) { acc += sampleMv(); delay(1); }
    center = acc / 12.0f;
    filt = center;
  }

  void begin(uint8_t p, bool inv) {
    pin = p; invert = inv;
    pinMode(pin, INPUT);
    recenter();
    lo = center - JOY_INIT_SPAN_MV;
    hi = center + JOY_INIT_SPAN_MV;
  }

  // returns -1..+1
  float update() {
    float mv  = sampleMv();
    float err = mv - filt;
    float t   = (fabsf(err) - JOY_NOISE_MV) / (JOY_FAST_MV - JOY_NOISE_MV);
    t = constrain(t, 0.0f, 1.0f);
    filt += (JOY_ALPHA_MIN + (JOY_ALPHA_MAX - JOY_ALPHA_MIN) * t) * err;

    if (filt < lo) lo = filt;
    if (filt > hi) hi = filt;

    float d    = filt - center;
    float span = (d >= 0) ? (hi - center) : (center - lo);
    if (span < JOY_MIN_SPAN_MV) span = JOY_MIN_SPAN_MV;
    float n = constrain(d / span, -1.0f, 1.0f);

    float a = fabsf(n);
    if (a < JOY_DEADZONE) return 0.0f;
    a = (a - JOY_DEADZONE) / (1.0f - JOY_DEADZONE);        // 0..1 after dead-zone
    a = a * ((1.0f - JOY_EXPO) + JOY_EXPO * a);            // expo blend
    n = (n < 0) ? -a : a;
    return invert ? -n : n;
  }
};
static JoyAxis joyX, joyY;

// =====================================================================
//  Button (debounced, short = toggle mode, long = reset + recentre)
// =====================================================================
static void resetView() { g_yaw = 0; g_pitch = 0; g_zoom = 1; }

static void pollButton() {
  static bool lastRaw = false, down = false, longFired = false;
  static uint32_t edgeT = 0, downT = 0;
  const bool raw = (digitalRead(PIN_JOY_SW) == LOW);
  const uint32_t now = millis();
  if (raw != lastRaw) { lastRaw = raw; edgeT = now; }
  if ((now - edgeT) >= DEBOUNCE_MS && raw != down) {
    down = raw;
    if (down) { downT = now; longFired = false; }
    else if (!longFired) { g_mode = (g_mode == MODE_ORBIT) ? MODE_ZOOM : MODE_ORBIT; }
  }
  if (down && !longFired && (now - downT) >= LONG_PRESS_MS) {
    longFired = true;
    resetView();
    joyX.recenter();
    joyY.recenter();
    g_flashUntil = now + 700;
  }
}

// =====================================================================
//  Network (runs on core 0)
// =====================================================================
// BLE frame receiver: each 128-byte application frame is sent as eight
// 20-byte GATT writes, compatible with the default ATT MTU (23).
// Packet [0xD7, sequence, chunkIndex, chunkCount=8, 16 data bytes].
// No-hand packet [0xD7, sequence, 0xFE, 0x00].
static void commitReceivedFrame() {
  const uint32_t now = millis();
  portENTER_CRITICAL(&g_mux);
  if (g_bleAssembly[0] == 0xA5 && (g_bleAssembly[1] & 1)) {
    memcpy(g_raw, g_bleAssembly + 2, sizeof(g_raw));
    g_present = true;
  } else {
    g_present = false;
  }
  g_lastRxMs = now;
  g_seq++;
  portEXIT_CRITICAL(&g_mux);
}

class BleRxCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* characteristic, NimBLEConnInfo& connInfo) override {
    const std::string value = characteristic->getValue();
    const size_t len = value.size();
    if (len < 4) return;
    const uint8_t* b = reinterpret_cast<const uint8_t*>(value.data());
    if (b[0] != BLE_PACKET_MAGIC) return;
    if (b[2] == 0xFE && b[3] == 0x00) {
      portENTER_CRITICAL(&g_mux);
      g_present = false;
      g_lastRxMs = millis();
      g_seq++;
      portEXIT_CRITICAL(&g_mux);
      return;
    }
    if (len != 4 + BLE_CHUNK_BYTES || b[3] != BLE_CHUNK_COUNT || b[2] >= BLE_CHUNK_COUNT) return;
    const uint8_t seq = b[1], idx = b[2];
    if (!g_bleAssemblyStarted || seq != g_bleAssemblySeq || idx == 0) {
      g_bleAssemblySeq = seq;
      g_bleAssemblyMask = 0;
      g_bleAssemblyStarted = true;
    }
    memcpy(g_bleAssembly + (size_t)idx * BLE_CHUNK_BYTES, b + 4, BLE_CHUNK_BYTES);
    g_bleAssemblyMask |= (uint8_t)(1U << idx);
    if (g_bleAssemblyMask == 0xFF) {
      commitReceivedFrame();
      g_bleAssemblyMask = 0;
      g_bleAssemblyStarted = false;
    }
  }
} bleRxCallbacks;

class BleServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* server, NimBLEConnInfo& connInfo) override {
    g_bleConnected = true;
    server->updateConnParams(connInfo.getConnHandle(), 18, 36, 0, 300);
  }
  void onDisconnect(NimBLEServer* server, NimBLEConnInfo& connInfo, int reason) override {
    g_bleConnected = false;
    g_bleAssemblyStarted = false;
    g_bleAssemblyMask = 0;
    NimBLEDevice::startAdvertising();
  }
} bleServerCallbacks;

static void startBluetooth() {
  NimBLEDevice::init(BLE_DEVICE_NAME);
  NimBLEDevice::setPower(-3);            // lower TX power for a nearby controller
  NimBLEDevice::setMTU(23);              // packet format is safe for the default MTU
  g_bleServer = NimBLEDevice::createServer();
  g_bleServer->setCallbacks(&bleServerCallbacks);
  NimBLEService* service = g_bleServer->createService(BLE_SERVICE_UUID);
  g_bleRx = service->createCharacteristic(BLE_RX_UUID,
      NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR, 20);
  g_bleRx->setCallbacks(&bleRxCallbacks);
  service->start();
  NimBLEAdvertising* advertising = NimBLEDevice::getAdvertising();
  advertising->setName(BLE_DEVICE_NAME);
  advertising->addServiceUUID(BLE_SERVICE_UUID);
  advertising->enableScanResponse(true);
  advertising->start();
  Serial.println("[BLE] advertising as ESP32-Hand-Mirror; Wi-Fi is not started");
}

// =====================================================================
//  OLED helpers
// =====================================================================
static void oledMsg(const char* a, const char* b = nullptr, const char* c = nullptr) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);  display.println(a);
  if (b) { display.setCursor(0, 16); display.println(b); }
  if (c) { display.setCursor(0, 32); display.println(c); }
  display.display();
}

// =====================================================================
//  Rendering
// =====================================================================
static void drawHand(float yaw, float pitch, float zoom) {
  const float cy = cosf(yaw), sy = sinf(yaw), cp = cosf(pitch), sp = sinf(pitch);
  const float S  = HAND_PX_PER_UNIT * zoom;
  const float CX = OLED_W * 0.5f, CY = HUD_H + (OLED_H - HUD_H) * 0.5f;
  float px[21], py[21], pz[21];

  for (uint8_t i = 0; i < 21; i++) {
    const float x = g_cur[i * 3], y = g_cur[i * 3 + 1], z = g_cur[i * 3 + 2];
    const float x1 = x * cy + z * sy;
    const float z1 = -x * sy + z * cy;
    const float y2 = y * cp - z1 * sp;
    const float z2 = y * sp + z1 * cp;
    float den = CAM_DIST - z2; if (den < 0.5f) den = 0.5f;
    const float k = CAM_DIST / den;                        // perspective
    px[i] = CX + x1 * S * k;
    py[i] = CY - y2 * S * k;
    pz[i] = z2;
  }

  // bones: nearer bones are drawn 2 px thick = cheap depth cue
  for (uint8_t b = 0; b < 21; b++) {
    const uint8_t i = BONES[b][0], j = BONES[b][1];
    const int x0 = (int)(px[i] + 0.5f), y0 = (int)(py[i] + 0.5f);
    const int x1 = (int)(px[j] + 0.5f), y1 = (int)(py[j] + 0.5f);
    display.drawLine(x0, y0, x1, y1, SSD1306_WHITE);
    if ((pz[i] + pz[j]) * 0.5f > 0.03f) display.drawLine(x0 + 1, y0, x1 + 1, y1, SSD1306_WHITE);
  }
  // joints
  for (uint8_t i = 0; i < 21; i++) {
    const int x = (int)(px[i] + 0.5f), y = (int)(py[i] + 0.5f);
    const bool tip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
    if (i == 0)       display.fillCircle(x, y, 2, SSD1306_WHITE);
    else if (tip)     display.fillCircle(x, y, (pz[i] > 0.0f) ? 2 : 1, SSD1306_WHITE);
    else              display.drawPixel(x, y, SSD1306_WHITE);
  }
}

static void drawHud(const char* status) {
  display.fillRect(0, 0, OLED_W, HUD_H, SSD1306_BLACK);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  char buf[32];
  snprintf(buf, sizeof(buf), "%3dfps %s %s", g_fps, (g_mode == MODE_ORBIT) ? "ORBIT" : "ZOOM ", status);
  display.print(buf);
}

static void drawInfoScreen(const char* line) {
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);   display.print("ESP32 3D HAND");
  display.setCursor(0, 12);  display.print("BLE "); display.print(g_bleConnected ? "CONNECTED" : "ADVERTISING");
  display.setCursor(0, 22);  display.print("Controller: BLE");
  display.setCursor(0, 36);  display.print(line);
  display.setCursor(0, 56);
  char buf[24];
  snprintf(buf, sizeof(buf), "%3dfps %s", g_fps, (g_mode == MODE_ORBIT) ? "ORBIT" : "ZOOM");
  display.print(buf);
}

// =====================================================================
//  setup / loop
// =====================================================================
void setup() {
  Serial.begin(115200);
  pinMode(PIN_JOY_SW, INPUT_PULLUP);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL);
  Wire.setClock(I2C_HZ);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR, true, false)) {
    Serial.println("SSD1306 not found - check wiring / address");
    for (;;) delay(1000);
  }
#if OLED_FAST_OSC
  display.ssd1306_command(SSD1306_SETDISPLAYCLOCKDIV);
  display.ssd1306_command(0xF0);
#endif

  oledMsg("Calibrating stick", "Release joystick", "Sampling X...");
  delay(120);
  joyX.begin(PIN_JOY_X, JOY_INVERT_X);
  oledMsg("Calibrating stick", "Release joystick", "Sampling Y...");
  joyY.begin(PIN_JOY_Y, JOY_INVERT_Y);

  oledMsg("Starting Bluetooth", "No Wi-Fi required");
  startBluetooth();
  oledMsg("BLE advertising", "Open controller page", "Select ESP32-Hand-Mirror");
  delay(700);
  g_lastUs = micros();
}

void loop() {                                              // rendering on core 1
  static bool wasLive = false;
  static uint32_t lastRxLocal = 0, fpsT = 0, fpsN = 0;
  static bool presentLocal = false;

  const uint32_t nowUs = micros();
  float dt = (uint32_t)(nowUs - g_lastUs) * 1e-6f;
  g_lastUs = nowUs;
  if (dt > 0.1f) dt = 0.1f;
  if (dt < 1e-4f) dt = 1e-4f;

  // ---- input
  pollButton();
  const float jx = joyX.update();
  const float jy = joyY.update();

  float yaw = g_yaw, pitch = g_pitch, zoom = g_zoom;
  yaw += jx * YAW_RATE * dt;
  if (yaw >  PI) yaw -= TWO_PI;
  if (yaw < -PI) yaw += TWO_PI;
  if (g_mode == MODE_ORBIT) {
    pitch = constrain(pitch + jy * PITCH_RATE * dt, -PITCH_LIMIT, PITCH_LIMIT);
  } else {
    zoom = constrain(zoom * expf(jy * ZOOM_RATE * dt), ZOOM_MIN, ZOOM_MAX);
  }
  g_yaw = yaw; g_pitch = pitch; g_zoom = zoom;

  // ---- new hand frame?
  const uint32_t seq = g_seq;
  if (seq != g_lastSeq) {
    int16_t tmp[63];
    portENTER_CRITICAL(&g_mux);
    memcpy(tmp, g_raw, sizeof(tmp));
    presentLocal = g_present;
    lastRxLocal  = g_lastRxMs;
    portEXIT_CRITICAL(&g_mux);
    g_lastSeq = seq;
    if (presentLocal) for (uint8_t k = 0; k < 63; k++) g_tgt[k] = tmp[k] * (1.0f / 16384.0f);
  }
  const bool ctrlAlive = (millis() - lastRxLocal) < STALE_MS && lastRxLocal != 0;
  const bool live = ctrlAlive && presentLocal;
  if (live && !wasLive) memcpy(g_cur, g_tgt, sizeof(g_cur));   // snap on (re)acquire
  wasLive = live;

  // ---- render
  display.clearDisplay();
  if (live) {
    const float a = 1.0f - expf(-dt * 1000.0f / HAND_SMOOTH_TAU_MS);   // frame-rate independent smoothing
    for (uint8_t k = 0; k < 63; k++) g_cur[k] += (g_tgt[k] - g_cur[k]) * a;
    drawHand(yaw, pitch, zoom);
    drawHud((millis() < g_flashUntil) ? "RESET" : "LIVE");
  } else {
    drawInfoScreen(ctrlAlive ? "No hand in view" : (g_bleConnected ? "Waiting for camera" : "Pair controller BLE"));
  }
  display.display();                                       // the I2C transfer dominates frame time

  // ---- fps counter
  fpsN++;
  const uint32_t ms = millis();
  if (ms - fpsT >= 500) { g_fps = (int)(fpsN * 1000UL / (ms - fpsT)); fpsN = 0; fpsT = ms; }
}
