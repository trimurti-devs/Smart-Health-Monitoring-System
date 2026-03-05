# Smart Health Monitoring System

A comprehensive IoT-based health monitoring platform designed to continuously track vital signs and detect critical health events using wearable sensors and edge computing. This system combines multiple physiological measurements with advanced signal processing algorithms to provide real-time health insights and emergency alerting capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Hardware Components](#hardware-components)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Sensor Specifications](#sensor-specifications)
- [Data Processing](#data-processing)
- [API Integration](#api-integration)
- [Safety & Alerts](#safety--alerts)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The Smart Health Monitoring System is an advanced embedded healthcare solution that integrates multiple sensors on ESP32 microcontrollers to capture comprehensive health metrics in real time. The system processes physiological signals using sophisticated signal processing algorithms and transmits data to a backend server for analysis, storage, and visualization.

**Key Capabilities:**
- **Real-time Vital Sign Monitoring**: Heart rate, SpO2, ECG, temperature
- **Motion Detection**: Falls, immobility, and unusual movement patterns
- **Muscle Activity Monitoring**: Electromyography (EMG) signals
- **Advanced HRV Analysis**: Heart Rate Variability with spectral and nonlinear metrics
- **Edge Processing**: On-device FFT and statistical analysis
- **Remote Alerting**: Network-based notifications for critical events
- **Multi-device Support**: Scalable to multiple patients simultaneously

---

## 🏗️ System Architecture

The system follows a distributed architecture with edge computing capabilities:

```
┌──────────────────────────────────────────────────────────────┐
│                    Smart Health Monitoring System             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐           ┌─────────────────┐          │
│  │   ESP32 Unit 1  │           │   ESP32 Unit 2  │          │
│  │ (ECG/PPG/HRV)   │           │ (Gyro/EMG/Temp) │          │
│  └────────┬────────┘           └────────┬────────┘          │
│           │                             │                    │
│           └──────────────┬──────────────┘                    │
│                          │                                   │
│                    WiFi Network                             │
│                          │                                   │
│           ┌──────────────┴──────────────┐                   │
│           │                             │                   │
│      ┌────▼──────┐            ┌────────▼────┐              │
│      │  Flask    │            │  Database   │              │
│      │  Backend  │────────────│  (Storage)  │              │
│      └────┬──────┘            └────────┬────┘              │
│           │                            │                    │
│      ┌────▼─────────────────────────────▼────┐             │
│      │   Dashboard / Analytics / Alerting     │             │
│      └──────────────────────────────────────┘             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Components:
- **Hardware Layer**: Sensor nodes (ESP32 + multiple sensors)
- **Edge Processing**: On-device signal processing and analysis
- **Communication**: WiFi-based data transmission (HTTP/JSON)
- **Backend**: Flask server for data reception and aggregation
- **Storage**: Database for historical data and analytics

---

## ✨ Features

### 1. **Cardiac & Respiratory Monitoring** (ECG-PPG-Stress.ino)
- **ECG Acquisition**: Direct electrocardiogram signals from AD8232 amplifier
- **PPG (Photoplethysmography)**: Pulse oximetry using MAX30100 sensor
- **Heart Rate Detection**: Real-time R-peak detection
- **Oxygen Saturation (SpO2)**: Blood oxygen level monitoring

### 2. **Advanced HRV Analysis**
The system computes extensive Heart Rate Variability metrics for stress and autonomic nervous system assessment:

#### Time-Domain Metrics:
- **Mean RR & Median RR**: Central tendency of cardiac intervals
- **SDRR**: Standard Deviation of RR intervals (overall variability)
- **RMSSD**: Root Mean Square of Successive Differences (parasympathetic activity)
- **SDSD**: Standard Deviation of Successive Differences
- **pNN25/pNN50**: Percentage of successive intervals differing by >25/50ms
- **Kurtosis & Skewness**: Distribution shape analysis

#### Frequency-Domain Metrics:
- **VLF Power** (0.003-0.04 Hz): Very Low Frequency (regulatory influence)
- **LF Power** (0.04-0.15 Hz): Low Frequency (sympathetic + parasympathetic)
- **HF Power** (0.15-0.4 Hz): High Frequency (parasympathetic dominance)
- **LF/HF Ratio**: Sympatho-vagal balance indicator
- **Spectral Normalization**: LF_nu, HF_nu (normalized units)

#### Nonlinear Metrics:
- **SD1 & SD2**: Poincaré plot descriptors
- **Sample Entropy**: Signal complexity measurement
- **Higuchi Fractal Dimension**: Self-similarity characterization

### 3. **Motion & Fall Detection** (Gyro-EMG-Temp.ino)
- **6-Axis IMU**: Accelerometer + Gyroscope from MPU6050
- **Free Fall Detection**: Identifies zero-gravity conditions
- **Impact Detection**: Recognizes high acceleration impacts
- **Tilt Angle Analysis**: Body position relative to gravity
- **Immobility Detection**: Alerts on prolonged unconsciousness

### 4. **Vital Signs Monitoring**
- **Temperature Sensing**: NTC thermistor with Steinhart-Hart equation
- **EMG Monitoring**: Electromyography for muscle activity assessment
- **Lead-off Detection**: Ensures electrode contact integrity (ECG)

### 5. **Intelligent Alerting**
- **Fall Confirmation Algorithm**: Multi-stage detection with 2-second validation
- **Immobility Alerts**: 5-second no-movement detection post-fall
- **Automatic State Reset**: 30-second timeout to prevent alert fatigue
- **Real-time Serial Debugging**: Emoji-annotated status messages

---

## 🔧 Hardware Components

### ESP32 Node 1: Cardiac & Stress Monitoring
| Component | Model | Function | Pins |
|-----------|-------|----------|------|
| Microcontroller | ESP32 | Main processing | - |
| ECG Amplifier | AD8232 | Single-lead ECG | GPIO 33 (ADC), 35, 32 |
| Pulse Oximeter | MAX30100 | PPG + SpO2 | I2C (SDA/SCL) |
| Communication | WiFi Module | Network | Built-in |

### ESP32 Node 2: Motion & Environmental Monitoring
| Component | Model | Function | Pins |
|-----------|-------|----------|------|
| Microcontroller | ESP32 | Main processing | - |
| IMU Sensor | MPU6050 | 6-axis motion | I2C (GPIO 21/22) |
| EMG Sensor | Generic | Muscle signals | GPIO 34 (ADC) |
| Thermistor | NTC 10kΩ | Temperature | GPIO 36 (ADC) |
| Communication | WiFi Module | Network | Built-in |

---

## 📁 Project Structure

```
Smart-Health-Monitoring-System/
├── Arduino/
│   ├── ECG-PPG-Stress.ino          # Cardiac monitoring & HRV analysis
│   └── Gyro-EMG-Temp.ino           # Motion & environmental monitoring
├── VS Code/                         # Development configuration
└── README.md                        # This file
```

### File Descriptions:

#### `ECG-PPG-Stress.ino` (730 lines)
**Purpose**: Comprehensive cardiac and stress monitoring

**Key Functions**:
- `setup()`: Initializes WiFi, MAX30100 sensor, serial communication
- `loop()`: Main acquisition and processing loop
- `interpolateRR()`: Linear interpolation of RR intervals for uniform sampling
- `calculatePSD()`: Power Spectral Density using Hamming-windowed FFT
- `bandPower()`: Computes frequency band powers (VLF/LF/HF)
- `calculateSD1/SD2()`: Poincaré plot metrics
- `calculateSampen()`: Sample entropy computation
- `calculateHiguchi()`: Fractal dimension analysis
- Plus 20+ statistical calculation functions

**Data Output (JSON)**:
- HR, SpO2, Systolic/Diastolic BP (PTT-based)
- 30+ HRV metrics (time, frequency, nonlinear domains)
- ECG raw values

**Transmission**: HTTP POST to Flask backend every ~1 second

---

#### `Gyro-EMG-Temp.ino` (218 lines)
**Purpose**: Motion detection, fall alerting, and environmental monitoring

**Key Functions**:
- `setup()`: Initializes I2C, MPU6050, WiFi connection
- `loop()`: Sensor acquisition and fall detection logic
- **Fall Detection Algorithm**:
  1. **Free Fall** (Stage 1): Acceleration < 0.5g for <500ms
  2. **Impact** (Stage 2): Acceleration jump to >2.5g
  3. **Tilt** (Stage 3): Body angle >80° for >2 seconds
  4. **Immobility** (Stage 4): <1000 raw units motion for >5 seconds

**Data Output (JSON)**:
- Device ID, Timestamp
- EMG, Temperature, Accelerometer (x,y,z), Gyroscope (x,y,z)
- Max magnitudes: acc_max, gyro_max, lin_max
- Fall & immobility detection flags

**Features**:
- Robust WiFi reconnection logic
- JSON serialization with ArduinoJson library
- Sensor validation and error handling
- Detailed serial debugging with emojis

---

## 🚀 Installation & Setup

### Prerequisites
- **Arduino IDE** (1.8.x or higher)
- **ESP32 Board Package** (via Arduino IDE Board Manager)
- **Libraries**:
  ```
  - Wire.h (I2C)
  - WiFi.h (WiFi)
  - HTTPClient.h (HTTP requests)
  - ArduinoJson.h (JSON serialization)
  - MAX30100_PulseOximeter.h (MAX30100 sensor)
  - MPU6050.h (IMU sensor)
  - arduinoFFT.h (FFT computation)
  ```

### Library Installation

In Arduino IDE:
1. **Sketch** → **Include Library** → **Manage Libraries**
2. Search for and install:
   - `ArduinoJson` by Benoit Blanchon
   - `MAX30100lib` by OXullo Intersecans
   - `MPU6050` by Electronic Cats
   - `arduinoFFT` by kosme

### Hardware Setup

#### ESP32 Node 1 (ECG-PPG):
```
MAX30100 Connections:
  - VCC → 3.3V
  - GND → GND
  - SDA → GPIO 21
  - SCL → GPIO 22
  - IRQ → (optional)

AD8232 ECG Connections:
  - +Vcc → 3.3V
  - GND → GND
  - OUTPUT → GPIO 33 (ADC1_CH5)
  - LO+ → GPIO 35 (Digital)
  - LO- → GPIO 32 (Digital)
```

#### ESP32 Node 2 (Motion-EMG-Temp):
```
MPU6050 Connections:
  - VCC → 3.3V
  - GND → GND
  - SDA → GPIO 21
  - SCL → GPIO 22
  - INT → (optional)

EMG Sensor:
  - Signal → GPIO 34 (ADC1_CH6)
  - GND → GND

NTC Thermistor (voltage divider):
  - Vcc (3.3V) → 10kΩ → ADC → Thermistor → GND
  - ADC Input → GPIO 36 (ADC1_CH0)
```

### Firmware Flashing

1. **Select Board**: Tools → Board → ESP32 → ESP32 Dev Module
2. **Select Port**: Tools → Port → [Your Serial Port]
3. **Upload Speed**: 115200 baud
4. **Flash**: Sketch → Upload
5. **Monitor**: Tools → Serial Monitor (115200 baud)

---

## ⚙️ Configuration

### WiFi Configuration

**File: Both .ino files**

```cpp
// Update these credentials
const char* ssid = "Your_WiFi_SSID";
const char* password = "Your_WiFi_Password";
const char* serverUrl = "http://192.168.1.X:5000/receive_data";
```

Replace `192.168.1.8` with your backend server IP address.

### Device Identification

**ECG-PPG-Stress.ino** (lines 44-47):
```cpp
String uuid = "patient_uuid_123";      // Unique patient identifier
String datasetId = "dataset_id_001";   // Dataset group identifier
String condition = "healthy";          // Patient condition: healthy/diseased/risk
```

**Gyro-EMG-Temp.ino** (line 17):
```cpp
const char* deviceId = "ESP32_1";  // Change to "ESP32_2" for second device
```

### Sensor Parameters

**HRV FFT Configuration** (ECG-PPG-Stress.ino, line 16):
```cpp
float samplingFrequency = 4.0;  // Hz - for RR interval spectral analysis
```

**Fall Detection Thresholds** (Gyro-EMG-Temp.ino, lines 98-134):
```cpp
// Free fall threshold
if (A_total < 0.5)  // Acceleration < 0.5g

// Impact threshold
if (A_total > 2.5)  // Acceleration > 2.5g

// Tilt threshold
if (abs(angle) > 80)  // Body angle > 80°

// Immobility threshold
if (abs(ax) < 1000)   // Very low acceleration
```

---

## 📊 Usage

### Starting the System

1. **Upload firmware** to both ESP32 devices
2. **Open Serial Monitor** (115200 baud) on each device
3. **Verify connections**:
   ```
   ✅ WiFi connected
   ✅ MAX30100 connected (Node 1) / MPU6050 connected (Node 2)
   ```
4. **Check backend server** is running and accessible
5. **Attach sensors** to patient and verify readings:
   ```
   ECG Node: Mean RR: xxx | HR: xx | SpO2: xx%
   Motion Node: EMG: xxxx | Temp: xx.xC | Acc: x.xx,x.xx,x.xx
   ```

### Real-Time Monitoring

**Serial Output Examples**:

ECG Node (every 1 second):
```
Mean RR: 850
Median RR: 845
SDRR: 42
RMSSD: 38
SDSD: 29
pNN50: 15
VLF Power: 125.3
LF Power: 542.8
HF Power: 298.4
LF/HF: 1.82
SD1: 27.0
SD2: 58.5
Sample Entropy: 1.34
```

Motion Node (every 1 second):
```
EMG: 2048 | Temp: 36.50°C | Acc: 0.98,0.05,0.12 | Gyro: 1.2,0.8,-0.3
📤 Sending JSON:
{"device_id":"ESP32_1","timestamp":45230,"emg":2048,"temperature":36.50,
 "accelerometer":{"x":0.98,"y":0.05,"z":0.12},"fall_detected":0}
✅ Server response: OK
```

### Data Interpretation

#### HRV Metrics:
- **High RMSSD/HF**: Parasympathetic dominance (relaxation)
- **High LF/HF ratio**: Sympathetic dominance (stress)
- **Low Sample Entropy**: Regular patterns (health)
- **High Higuchi Dimension**: Complex dynamics (stress/disease)

#### Fall Detection:
- **Multi-stage confirmation** reduces false positives
- **Immobility alert** suggests unconsciousness
- **30-second reset** prevents alert spam

---

## 📈 Sensor Specifications

### MAX30100 Pulse Oximeter
| Parameter | Value |
|-----------|-------|
| Operating Voltage | 1.8 - 3.3V |
| SpO2 Accuracy | ±2% (70-100%) |
| HR Range | 30 - 255 bpm |
| Sampling Rate | 100 Hz (programmable) |
| I2C Address | 0x57 |
| LED Current | 0-50mA (programmable) |

### AD8232 ECG Amplifier
| Parameter | Value |
|-----------|-------|
| Supply Voltage | 3.3V single supply |
| Gain | ~100-1000 V/V |
| BW | 0.5-40 Hz typical |
| Output Range | 0-3.3V |
| Input Impedance | >10^9 Ω |

### MPU6050 IMU
| Parameter | Value |
|-----------|-------|
| 3-Axis Accelerometer | ±2g to ±16g (configurable) |
| 3-Axis Gyroscope | ±250 to ±2000 °/s (configurable) |
| Sampling Rate | 1kHz |
| I2C Address | 0x68/0x69 |
| Digital Filter | Programmable DMP |

---

## 🔬 Data Processing

### Signal Conditioning Pipeline

```
Raw Signal → Low-Pass Filter → Normalization → Peak Detection → Analysis
```

### HRV Computation Flow

1. **R-Peak Detection**: Threshold-based ECG signal processing
2. **RR Interval Extraction**: Timestamp differences between consecutive beats
3. **Statistical Analysis**:
   - Time-domain: Mean, SD, percentiles
   - Frequency-domain: FFT → PSD → Band Powers
   - Nonlinear: Poincaré plots, entropy, fractal dimension

### FFT Configuration
```cpp
#define FFT_SIZE 128
float samplingFrequency = 4.0 Hz;
Window Function: Hamming
Resolution: 4.0 / 128 = 0.03125 Hz/bin
```

---

## 🌐 API Integration

### Data Format (JSON)

#### ECG-PPG Node Example:
```json
{
  "uuid": "patient_uuid_123",
  "datasetId": "dataset_id_001",
  "condition": "healthy",
  "hr": 72,
  "spo2": 98,
  "sys": 118,
  "dia": 78,
  "meanRR": 850,
  "medianRR": 845,
  "sdr": 42,
  "rmssd": 38,
  "sdsd": 29,
  "pnn25": 22,
  "pnn50": 15,
  "kurt": 0.85,
  "skew": 0.12,
  "vlf": 125.3,
  "lf": 542.8,
  "hf": 298.4,
  "lf_nu": 64.5,
  "hf_nu": 35.5,
  "lf_hf": 1.82,
  "tp": 966.5,
  "sd1": 27.0,
  "sd2": 58.5,
  "sampen": 1.34,
  "higuchi": 1.25,
  "ecg": 1856
}
```

#### Motion-EMG Node Example:
```json
{
  "device_id": "ESP32_1",
  "timestamp": 45230,
  "emg": 2048,
  "temperature": 36.50,
  "accelerometer": {
    "x": 0.98,
    "y": 0.05,
    "z": 0.12
  },
  "gyroscope": {
    "x": 1.2,
    "y": 0.8,
    "z": -0.3
  },
  "acc_max": 1.00,
  "gyro_max": 1.20,
  "lin_max": 0.12,
  "fall_detected": 0,
  "immobility_detected": 0
}
```

### Backend Requirements

**Endpoint**: `POST /receive_data`

**Expected Response**: JSON with status
```json
{"status": "OK", "message": "Data received successfully"}
```

---

## ⚠️ Safety & Alerts

### Fall Detection Algorithm

The system implements a **4-stage confirmation** process to minimize false positives:

| Stage | Condition | Timeout | Action |
|-------|-----------|---------|--------|
| 1 | Acceleration < 0.5g | - | Set flag |
| 2 | Acceleration > 2.5g | <500ms | Detect impact |
| 3 | Tilt angle > 80° | <2s | Confirm fall |
| 4 | Immobility < 1000 units | >5s | Alert unconsciousness |

### Critical Alerts

1. **Fall Confirmed**: Automatic after Stage 3
2. **Unconsciousness Alert**: After Stage 4 (5+ seconds immobility)
3. **Sensor Disconnection**: Lead-off detection (ECG) / WiFi loss
4. **ECG Freeze Recovery**: Automatic MAX30100 reinitiation

### Safety Considerations

- **No medical diagnosis**: System provides monitoring only
- **Manual verification**: All alerts require human confirmation
- **Local deployment**: WiFi should be dedicated/secure
- **Data privacy**: Implement authentication/encryption for production
- **Calibration**: BP estimation is uncalibrated (PTT-based placeholder)

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

1. **Machine Learning**: Stress/disease classification models
2. **Cloud Integration**: AWS/Azure backend services
3. **Mobile App**: Real-time visualization dashboard
4. **Hardware**: Additional sensors (temperature, humidity)
5. **Signal Processing**: Advanced filtering, artifact removal
6. **Documentation**: Setup guides, troubleshooting

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 📞 Support & Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| WiFi connection fails | Check SSID/password, verify router is 2.4GHz |
| MAX30100 not detected | Verify I2C pull-ups (4.7kΩ), check SDA/SCL pins |
| MPU6050 not detected | Same as above, ensure 0x68 I2C address |
| High false fall alerts | Calibrate acceleration thresholds for user activity |
| Garbled serial output | Verify baud rate is 115200, check USB cable |
| Server not receiving data | Confirm serverUrl is correct, test ping to server |
| Sensors freeze periodically | Check power supply, add 100µF capacitors near sensors |

### Debug Mode

Enable verbose serial output:
```cpp
Serial.println("DEBUG: Entering loop()");
Serial.printf("Acc: %.2f, Gyro: %.2f\n", acc_total, gyro_max);
```

---

## 📚 References

- **HRV Analysis**: Malik et al., Task Force Guidelines (1996)
- **Fall Detection**: IEEE Standards on Gait & Motion
- **FFT**: Cooley-Tukey Algorithm
- **Sample Entropy**: Richman & Moorman (2000)
- **Higuchi Dimension**: Fractal Analysis Methods

---

**Version**: 1.0  
**Last Updated**: 2026-03-05 14:03:35  
**Developed by**: Trimurti Developers