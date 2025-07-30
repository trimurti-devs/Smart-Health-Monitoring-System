#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <MPU6050.h>

// Pins
#define EMG_PIN 34         // EMG analog input
#define TEMP_PIN 36        // NTC Thermistor analog input (VP)

// WiFi credentials
const char* ssid = "Argha";
const char* password = "Argha123";
const char* serverUrl = "http://192.168.1.8:5000/receive_data";

// Device identifier - change this for each ESP32!
const char* deviceId = "ESP32_1";  // Change to "ESP32_2" on the second device

MPU6050 mpu;
WiFiClient client;
HTTPClient http;

// Fall detection states
bool freeFallDetected = false;
bool impactDetected = false;
bool confirmedFall = false;
bool immobilityDetected = false;
unsigned long fallStartTime = 0;
unsigned long immobilityStartTime = 0;

void setup() {
    Serial.begin(115200);
    Wire.begin(21, 22);  // SDA, SCL pins
    
    // Initialize MPU6050
    Serial.println("Initializing MPU6050...");
    mpu.initialize();
    
    if (!mpu.testConnection()) {
        Serial.println("❌ MPU6050 connection failed!");
        while (1);
    }
    Serial.println("✅ MPU6050 connected!");

    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n✅ Connected to WiFi!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Device ID: ");
    Serial.println(deviceId);
}

void loop() {
    // Reconnect WiFi if needed
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️ WiFi disconnected! Reconnecting...");
        WiFi.begin(ssid, password);
        delay(2000);
        return;
    }

    // Read EMG value
    int emgValue = analogRead(EMG_PIN);
    
    // Read temperature from NTC thermistor
    int rawTemp = analogRead(TEMP_PIN);
    float voltage = rawTemp * 3.3 / 4095.0;
    float resistance = (10000 * voltage) / (3.3 - voltage);
    float temperature = 1.0 / (log(resistance / 10000.0) / 3950.0 + 1 / 298.15) - 273.15;

    // Read MPU6050 sensor values (raw)
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw values to physical units
    float accX = ax / 16384.0; // Convert to g (±2g range)
    float accY = ay / 16384.0;
    float accZ = az / 16384.0;
    float gyroX = gx / 131.0;  // Convert to degrees per second (±250°/s range)
    float gyroY = gy / 131.0;
    float gyroZ = gz / 131.0;
    
    // Calculate total acceleration magnitude
    float A_total = sqrt(accX * accX + accY * accY + accZ * accZ);

    // Print sensor values for debugging
    Serial.printf("EMG: %d | Temp: %.2f°C | Acc: %.2f,%.2f,%.2f | Gyro: %.2f,%.2f,%.2f\n",
                emgValue, temperature, accX, accY, accZ, gyroX, gyroY, gyroZ);

    // Fall Detection Algorithm
    
    // 🛑 Free Fall Detection: Acceleration < 0.5g (≈4.9 m/s²)
    if (A_total < 0.5 && !freeFallDetected) {
        freeFallDetected = true;
        fallStartTime = millis();
        Serial.println("🛑 Free fall detected!");
    }

    // 💥 Impact Detection: Acceleration > 2.5g
    if (freeFallDetected && (millis() - fallStartTime < 500)) {
        if (A_total > 2.5) {
            impactDetected = true;
            Serial.println("💥 Severe impact detected!");
        }
    }

    // 🚨 Tilt Detection: Must be tilted >80° for 2+ seconds
    if (impactDetected) {
        float angle = atan2(accX, accZ) * (180.0 / PI);
        if (abs(angle) > 80) {
            if (!confirmedFall) {
                fallStartTime = millis();
                confirmedFall = true;
            }
        }

        if (confirmedFall && (millis() - fallStartTime > 2000)) {
            Serial.println("⚠️ FALL CONFIRMED!");
        }
    }

    // ⏳ Immobility Detection: No movement for 5+ seconds after impact
    if (confirmedFall && !immobilityDetected) {
        if (abs(ax) < 1000 && abs(ay) < 1000 && abs(az) < 1000) { // Very little motion
            if (immobilityStartTime == 0) {
                immobilityStartTime = millis();
            }
            if (millis() - immobilityStartTime > 5000) { // 5 seconds immobility
                Serial.println("🚨 POSSIBLE UNCONSCIOUSNESS DETECTED!");
                immobilityDetected = true;
            }
        } else {
            immobilityStartTime = 0; // Reset timer if movement is detected
        }
    }

    // Reset states if no fall
    if (A_total > 0.8 && !confirmedFall) {
        freeFallDetected = false;
        impactDetected = false;
    }

    // Reset fall detection after 30 seconds to prevent persistent alerts
    if (confirmedFall && (millis() - fallStartTime > 30000)) {
        confirmedFall = false;
        immobilityDetected = false;
        freeFallDetected = false;
        impactDetected = false;
        fallStartTime = 0;
        immobilityStartTime = 0;
        Serial.println("Fall detection reset");
    }

    // 📤 Prepare JSON Data for Server
    StaticJsonDocument<512> jsonDoc;
    
    // Add device identifier
    jsonDoc["device_id"] = deviceId;
    jsonDoc["timestamp"] = millis();
    
    // Add vital signs
    jsonDoc["emg"] = emgValue;
    jsonDoc["temperature"] = temperature;
    
    // Add accelerometer data in both formats for compatibility
    jsonDoc["x"] = accX;
    jsonDoc["y"] = accY;
    jsonDoc["z"] = accZ;
    
    // Nested objects for your dashboard
    JsonObject accel = jsonDoc.createNestedObject("accelerometer");
    accel["x"] = accX;
    accel["y"] = accY;
    accel["z"] = accZ;
    
    JsonObject gyroscope = jsonDoc.createNestedObject("gyroscope");
    gyroscope["x"] = gyroX;
    gyroscope["y"] = gyroY;
    gyroscope["z"] = gyroZ;
    
    // Add fall detection metrics
    jsonDoc["acc_max"] = A_total;
    jsonDoc["gyro_max"] = max(abs(gyroX), max(abs(gyroY), abs(gyroZ)));
    jsonDoc["lin_max"] = abs(accZ);
    jsonDoc["acc_skewness"] = abs(accX);  // Placeholder - calculate actual skewness if needed
    jsonDoc["gyro_skewness"] = abs(gyroZ); // Placeholder - calculate actual skewness if needed
    jsonDoc["fall_detected"] = (freeFallDetected || confirmedFall) ? 1 : 0;  
    jsonDoc["immobility_detected"] = immobilityDetected ? 1 : 0;  

    String jsonData;
    serializeJson(jsonDoc, jsonData);

    Serial.println("📤 Sending JSON:");
    Serial.println(jsonData);

    // 🌐 Send Data to Server
    http.begin(client, serverUrl);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonData);
    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("✅ Server response: " + response);
    } else {
        Serial.print("❌ HTTP Error: ");
        Serial.println(httpResponseCode);
    }
    http.end();

    delay(1000); // Send data every second
}
