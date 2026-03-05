# Smart Health Monitoring System

## Comprehensive Documentation

A state-of-the-art real-time health monitoring system that integrates multiple biometric sensors with advanced machine learning algorithms for intelligent fall detection and stress prediction. This system represents a complete end-to-end solution for continuous health monitoring, from sensor data acquisition to real-time visualization and AI-powered analysis.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Hardware Components](#hardware-components)
4. [Firmware Implementation](#firmware-implementation)
5. [Advanced HRV Analysis](#advanced-hrv-analysis)
6. [Fall Detection System](#fall-detection-system)
7. [Machine Learning Models](#machine-learning-models)
8. [Backend API](#backend-api)
9. [Web Dashboard](#web-dashboard)
10. [Installation Guide](#installation-guide)
11. [API Reference](#api-reference)
12. [Data Specifications](#data-specifications)
13. [Troubleshooting Guide](#troubleshooting-guide)

---

## Executive Summary

The Smart Health Monitoring System is designed to provide continuous, real-time health monitoring for patients, elderly individuals, or anyone requiring health surveillance. The system combines cutting-edge hardware sensors with sophisticated software algorithms to deliver actionable health insights.

### Core Capabilities

- **Cardiovascular Monitoring**: Continuous ECG and PPG monitoring with advanced Heart Rate Variability (HRV) analysis
- **Motion Analysis**: 6-axis inertial measurement with intelligent fall detection
- **Physiological Sensing**: Temperature and muscle activity monitoring
- **AI-Powered Analysis**: Machine learning models for stress assessment and fall prediction
- **Real-Time Visualization**: Web-based dashboard with live data streaming

---

## System Architecture

### High-Level Design

The system follows a distributed architecture with three primary tiers:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Web Dashboard (Browser)                 │   │
│  │  • Real-time charts and visualizations               │   │
│  │  • Alert notifications                               │   │
│  │  • Historical data logging                           │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ WebSocket / HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Flask Server (Python)                 │   │
│  │  • REST API endpoints                                │   │
│  │  • Machine learning inference                        │   │
│  │  • Real-time data processing                         │   │
│  │  • WebSocket event handling                          │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST
┌──────────────────────────▼──────────────────────────────────┐
│                    DATA ACQUISITION LAYER                    │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   ESP32 #1          │    │   ESP32 #2                  │ │
│  │  (ECG/PPG Module)   │    │  (Motion/EMG Module)        │ │
│  │  • AD8232 ECG       │    │  • MPU6050 IMU              │ │
│  │  • MAX30100 PPG     │    │  • MyoWare EMG              │ │
│  │  • HRV Analysis     │    │  • NTC Temperature          │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Communication Flow

1. **ESP32 Devices** continuously sample sensor data at specified intervals
2. **Data Transmission** occurs via HTTP POST requests to the Flask server
3. **Server Processing** includes data validation, ML inference, and WebSocket broadcasting
4. **Dashboard Updates** in real-time through SocketIO events
5. **Alert Generation** when anomalies are detected (falls, high stress, etc.)

---

## Hardware Components

### Microcontroller Platform

The system utilizes the **ESP32** microcontroller as the core processing unit for both sensor modules. The ESP32 was selected for its:

- **Dual-core Processing**: 240MHz clock speed enables real-time signal processing
- **Wireless Connectivity**: Integrated WiFi 802.11 b/g/n for seamless data transmission
- **Rich Peripheral Interface**: Multiple I2C, SPI, and ADC channels for sensor integration
- **Low Power Consumption**: Suitable for battery-operated wearable applications
- **Cost-Effectiveness**: Affordable yet powerful platform for IoT applications

### Sensor Specifications

#### 1. ECG Sensor (AD8232)

The AD8232 is a single-lead heart rate monitor front-end designed for fitness and wellness applications.

**Technical Specifications:**
- Operating Voltage: 2.0V - 3.5V
- Supply Current: 170μA (typical)
- Input Bias Current: 1pA
- Common-Mode Rejection Ratio: 80dB
- Gain: 1100 V/V

**Integration Details:**
- Connected to ESP32 GPIO33 (analog input)
- Lead-off detection via GPIO35 (LO+) and GPIO32 (LO-)
- Right-leg drive circuit for noise reduction
- High-pass filter: 0.5Hz cutoff for baseline wander removal
- Low-pass filter: 40Hz cutoff for muscle artifact suppression

#### 2. PPG Sensor (MAX30100)

The MAX30100 is a pulse oximeter and heart-rate sensor that combines LEDs and photodetectors.

**Technical Specifications:**
- LED Wavelength: Red (660nm) + IR (880nm)
- LED Current: Programmable (11mA - 50mA)
- SpO2 Accuracy: ±2% (70-100% range)
- Heart Rate Accuracy: ±3 BPM
- I2C Interface: 400kHz maximum

**Integration Details:**
- Connected via I2C (GPIO21 SDA, GPIO22 SCL)
- Sampling rate: 50Hz (20ms intervals)
- LED current set to 11mA for optimal finger detection
- Automatic gain control for varying skin tones
- Finger detection algorithm prevents false readings

#### 3. Inertial Measurement Unit (MPU6050)

The MPU6050 combines a 3-axis gyroscope and 3-axis accelerometer in a single package.

**Technical Specifications:**
- Accelerometer Range: ±2g, ±4g, ±8g, ±16g (configurable)
- Gyroscope Range: ±250°/s, ±500°/s, ±1000°/s, ±2000°/s
- ADC Resolution: 16-bit for both sensors
- I2C Address: 0x68 (AD0 grounded)
- Digital Motion Processor (DMP) for sensor fusion

**Integration Details:**
- Connected via I2C (GPIO21 SDA, GPIO22 SCL)
- Accelerometer configured for ±2g range
- Gyroscope configured for ±250°/s range
- Digital Low Pass Filter (DLPF) enabled at 44Hz
- Sampling rate: ~100Hz for motion analysis

#### 4. EMG Sensor (MyoWare Cable Shield)

The MyoWare Cable Shield allows for remote EMG electrode placement.

**Technical Specifications:**
- Operating Voltage: 3.3V - 5V
- Output Voltage: 0V - 3.3V
- Gain: 1000x
- Bandwidth: 10Hz - 500Hz
- Input Impedance: >1GΩ

**Integration Details:**
- Connected to ESP32 GPIO34 (analog input)
- Three-electrode configuration: Reference, Mid-Muscle, End-Muscle
- Raw EMG signal sampled at ~250Hz
- Signal envelope detection for muscle activation analysis

#### 5. Temperature Sensor (NTC Thermistor)

A 10kΩ NTC (Negative Temperature Coefficient) thermistor for body temperature measurement.

**Technical Specifications:**
- Resistance at 25°C: 10kΩ
- Beta Value: 3950K
- Temperature Range: -40°C to +125°C
- Accuracy: ±0.5°C (with calibration)

**Integration Details:**
- Connected to ESP32 GPIO36 (VP - analog input)
- Voltage divider circuit with 10kΩ reference resistor
- Steinhart-Hart equation for temperature conversion:
  ```
  T = 1 / (ln(R/R0)/B + 1/T0) - 273.15
  ```
  Where R0 = 10kΩ, B = 3950K, T0 = 298.15K

### Pin Configuration Summary

| Device | Pin | Function | Notes |
|--------|-----|----------|-------|
| **ECG-PPG** | GPIO33 | ECG Analog Input | AD8232 output |
| | GPIO35 | LO+ Detect | Lead-off positive |
| | GPIO32 | LO- Detect | Lead-off negative |
| | GPIO21 | I2C SDA | MAX30100 data |
| | GPIO22 | I2C SCL | MAX30100 clock |
| **Gyro-EMG** | GPIO34 | EMG Analog Input | MyoWare output |
| | GPIO36 | Temp Input | NTC thermistor |
| | GPIO21 | I2C SDA | MPU6050 data |
| | GPIO22 | I2C SCL | MPU6050 clock |

---

## Firmware Implementation

### ECG-PPG-Stress.ino

This firmware implements comprehensive cardiovascular monitoring with advanced signal processing.

#### Signal Processing Pipeline

**ECG Signal Chain:**
1. **Analog Acquisition**: 12-bit ADC sampling at ~250Hz
2. **Lead-off Detection**: Digital inputs monitor electrode contact
3. **R-peak Detection**: Threshold-based algorithm with 300ms refractory period
4. **RR Interval Calculation**: Timestamp difference between consecutive R-peaks
5. **HRV Metric Computation**: Real-time calculation of 30+ parameters

**PPG Signal Chain:**
1. **I2C Communication**: MAX30100 register reading at 50Hz
2. **Beat Detection**: Hardware interrupt on beat detection
3. **SpO2 Calculation**: Ratio of Ratios algorithm using red and IR signals
4. **Heart Rate**: Beat-to-beat interval measurement
5. **Signal Quality**: Finger detection via IR signal strength

#### Advanced HRV Algorithms

**Frequency Domain Analysis:**
```cpp
// FFT-based spectral analysis
void interpolateRR(const std::vector<long>& rrDiffs, 
                  std::vector<double>& interpolated, double fs) {
    // Linear interpolation to uniform sampling frequency
    // Required for FFT analysis of irregular RR intervals
}

void calculatePSD(const std::vector<double>& interpolated, 
                   std::vector<double>& psd) {
    // Hamming window application
    // Complex FFT computation
    // Magnitude squared for power spectrum
}

double bandPower(const std::vector<double>& psd, double fs, 
                 double lowFreq, double highFreq) {
    // Integration of power spectral density
    // Returns power in specified frequency band
}
```

**Nonlinear Analysis:**
```cpp
float calculateSampen(const std::vector<long>& rrIntervals) {
    // Sample Entropy algorithm
    // Parameters: m=2, r=0.2*SD
    // Measures signal complexity and regularity
}

float calculateHiguchi(const std::vector<long>& rrIntervals) {
    // Higuchi Fractal Dimension
    // kmax=5 for short time series
    // Measures self-similarity and fractal properties
}
```

**Poincaré Plot Analysis:**
```cpp
float calculateSD1(const std::vector<long>& rrIntervals) {
    // Short-term HRV from Poincaré plot
    // SD1 = sqrt(variance of RR[i+1] - RR[i]) / sqrt(2)
}

float calculateSD2(const std::vector<long>& rrIntervals) {
    // Long-term HRV from Poincaré plot
    // SD2 = sqrt(2*SDRR² - SD1²)
}
```

#### Blood Pressure Estimation

The firmware implements Pulse Transit Time (PTT) based blood pressure estimation:

```cpp
// PTT = Time difference between ECG R-peak and PPG peak
float PTT = (ppgPeakTime - rPeakTime) / 1000.0; // in seconds

// Simplified calibration model
float sysBP = 120 - (PTT * 50);  // Systolic
float diaBP = 80 - (PTT * 30);   // Diastolic
```

**Note:** This requires individual calibration for accurate absolute values but provides useful trending information.

### Gyro-EMG-Temp.ino

This firmware implements motion analysis, fall detection, and physiological monitoring.

#### Motion Processing

**Accelerometer Data Processing:**
```cpp
// Raw to physical units conversion
float accX = ax / 16384.0;  // Convert to g (±2g range)
float accY = ay / 16384.0;
float accZ = az / 16384.0;

// Total acceleration magnitude
float A_total = sqrt(accX*accX + accY*accY + accZ*accZ);
```

**Gyroscope Data Processing:**
```cpp
// Raw to physical units conversion
float gyroX = gx / 131.0;  // Convert to °/s (±250°/s range)
float gyroY = gy / 131.0;
float gyroZ = gz / 131.0;
```

#### 4-Stage Fall Detection Algorithm

**Stage 1: Free Fall Detection**
```cpp
// Monitor for weightlessness condition
if (A_total < 0.5 && !freeFallDetected) {
    freeFallDetected = true;
    fallStartTime = millis();
    // Log: "Free fall detected!"
}
```

**Stage 2: Impact Detection**
```cpp
// Look for high-g impact within 500ms of free fall
if (freeFallDetected && (millis() - fallStartTime < 500)) {
    if (A_total > 2.5) {
        impactDetected = true;
        // Log: "Severe impact detected!"
    }
}
```

**Stage 3: Tilt Detection**
```cpp
// Check if person remains tilted
if (impactDetected) {
    float angle = atan2(accX, accZ) * (180.0 / PI);
    if (abs(angle) > 80) {
        if (!confirmedFall) {
            fallStartTime = millis();
            confirmedFall = true;
        }
    }
    
    if (confirmedFall && (millis() - fallStartTime > 2000)) {
        // Log: "FALL CONFIRMED!"
    }
}
```

**Stage 4: Immobility Detection**
```cpp
// Monitor for lack of movement after fall
if (confirmedFall && !immobilityDetected) {
    if (abs(ax) < 1000 && abs(ay) < 1000 && abs(az) < 1000) {
        if (immobilityStartTime == 0) {
            immobilityStartTime = millis();
        }
        if (millis() - immobilityStartTime > 5000) {
            // Log: "POSSIBLE UNCONSCIOUSNESS!"
            immobilityDetected = true;
        }
    } else {
        immobilityStartTime = 0;  // Reset if movement detected
    }
}
```

#### Temperature Calculation

```cpp
// NTC thermistor reading and conversion
int rawTemp = analogRead(TEMP_PIN);
float voltage = rawTemp * 3.3 / 4095.0;
float resistance = (10000 * voltage) / (3.3 - voltage);

// Steinhart-Hart equation
float temperature = 1.0 / (log(resistance / 10000.0) / 3950.0 + 1 / 298.15) - 273.15;
```

---

## Advanced HRV Analysis

The system implements comprehensive Heart Rate Variability analysis across three domains:

### Time Domain Analysis

Time domain metrics quantify the variability in beat-to-beat intervals without frequency transformation.

| Metric | Full Name | Calculation | Clinical Interpretation |
|--------|-----------|-------------|------------------------|
| **meanRR** | Mean RR Interval | Average of all RR intervals | Baseline heart rate |
| **medianRR** | Median RR Interval | Middle value of sorted RR | Robust HR estimate |
| **SDRR** | Standard Deviation of RR | √(Σ(RRᵢ - mean)²/N) | Overall HRV magnitude |
| **RMSSD** | Root Mean Square of Successive Differences | √(Σ(RRᵢ₊₁ - RRᵢ)²/(N-1)) | Parasympathetic activity |
| **SDSD** | Standard Deviation of Successive Differences | SD of beat-to-beat differences | Short-term variability |
| **pNN25** | Percentage of NN50 | (Count >25ms / Total) × 100 | Fast HRV component |
| **pNN50** | Percentage of NN50 | (Count >50ms / Total) × 100 | Parasympathetic marker |
| **SDRR/RMSSD** | SDRR to RMSSD Ratio | SDRR / RMSSD | HRV balance indicator |

**Relative RR Metrics:**
The system also calculates normalized metrics to account for heart rate changes:
- meanRelRR, medianRelRR, SDRR_Rel_RR, RMSSD_Rel_RR, etc.

### Frequency Domain Analysis

Frequency domain analysis decomposes HRV into different frequency components using Fast Fourier Transform (FFT).

| Band | Frequency | Physiological Meaning | Normal Range |
|------|-----------|----------------------|--------------|
| **VLF** | 0.003-0.04 Hz | Very Low Frequency | 15-30% of total power |
| **LF** | 0.04-0.15 Hz | Low Frequency | 40-60% of total power |
| **HF** | 0.15-0.4 Hz | High Frequency | 20-40% of total power |

**Computed Metrics:**
- **VLF Power**: Raw power in VLF band
- **LF Power**: Raw power in LF band
- **HF Power**: Raw power in HF band
- **LF nu**: LF normalized units = LF / (Total - VLF) × 100
- **HF nu**: HF normalized units = HF / (Total - VLF) × 100
- **LF/HF Ratio**: Sympathovagal balance indicator
- **Total Power**: Sum of all frequency bands

**FFT Implementation:**
```cpp
// FFT parameters
#define FFT_SIZE 128
float vReal[FFT_SIZE];
float vImag[FFT_SIZE];
ArduinoFFT<float> fft;

// Interpolation to uniform sampling (4Hz)
interpolateRR(rrDiffs, interpolatedRR, 4.0);

// Windowing and FFT
fft.windowing(vReal, FFT_SIZE, FFTWindow::Hamming, FFTDirection::Forward);
fft.compute(vReal, vImag, FFT_SIZE, FFTDirection::Forward);
fft.complexToMagnitude(vReal, vImag, FFT_SIZE);
```

### Nonlinear Analysis

Nonlinear metrics capture complex patterns in heart rate variability that linear methods miss.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **SD1** | Poincaré plot short axis | Short-term HRV (parasympathetic) |
| **SD2** | Poincaré plot long axis | Long-term HRV (sympathetic + parasympathetic) |
| **Sample Entropy** | Measure of signal complexity | Lower = more regular, Higher = more complex |
| **Higuchi FD** | Fractal dimension | 1.0-2.0 range, higher = more fractal structure |
| **Kurtosis** | Distribution tail weight | >3 = heavy tails, <3 = light tails |
| **Skewness** | Distribution asymmetry | >0 = right-skewed, <0 = left-skewed |

**Poincaré Plot:**
A scatter plot of RR[i+1] vs RR[i] that reveals HRV patterns.
- SD1: Standard deviation perpendicular to line of identity
- SD2: Standard deviation along line of identity

**Sample Entropy Algorithm:**
```cpp
// Parameters: m = 2 (template length), r = 0.2 × SD (tolerance)
// Counts self-matches of length m and m+1
// Returns: -log(A/B) where A = matches of length m+1, B = matches of length m
```

---

## Fall Detection System

### Overview

The fall detection system implements a sophisticated 4-stage algorithm that progressively validates fall events to minimize false positives while ensuring rapid detection of genuine falls.

### Stage-by-Stage Analysis

#### Stage 1: Free Fall Detection

**Purpose:** Detect the initial weightlessness phase of a fall.

**Mechanism:**
- Continuously monitors total acceleration magnitude: A_total = √(ax² + ay² + az²)
- Trigger condition: A_total < 0.5g (4.9 m/s²)
- This indicates the person is in free fall (weightless condition)

**Implementation:**
```cpp
if (A_total < 0.5 && !freeFallDetected) {
    freeFallDetected = true;
    fallStartTime = millis();
    Serial.println("🛑 Free fall detected!");
}
```

**Rationale:** During a genuine fall, the person experiences brief weightlessness as they accelerate downward. This is the first reliable indicator of a fall event.

#### Stage 2: Impact Detection

**Purpose:** Confirm the fall with impact detection.

**Mechanism:**
- Monitors for high-g deceleration following free fall
- Time window: Must occur within 500ms of free fall detection
- Trigger condition: A_total > 2.5g (24.5 m/s²)

**Implementation:**
```cpp
if (freeFallDetected && (millis() - fallStartTime < 500)) {
    if (A_total > 2.5) {
        impactDetected = true;
        Serial.println("💥 Severe impact detected!");
    }
}
```

**Rationale:** A genuine fall concludes with impact. The 500ms window ensures we're detecting impact from the same fall event, not a separate activity.

#### Stage 3: Tilt Detection

**Purpose:** Verify the person remains in a fallen position.

**Mechanism:**
- Calculates tilt angle from accelerometer data: angle = atan2(accX, accZ) × (180/π)
- Trigger condition: |angle| > 80° for 2+ seconds
- Indicates person is lying down and unable to get up

**Implementation:**
```cpp
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
```

**Rationale:** Distinguishes falls from activities like jumping or rapid sitting. A genuine fall victim typically remains on the ground.

#### Stage 4: Immobility Detection

**Purpose:** Detect potential unconsciousness or inability to move.

**Mechanism:**
- Monitors for minimal motion across all accelerometer axes
- Trigger condition: |ax| < 1000 AND |ay| < 1000 AND |az| < 1000 for 5+ seconds
- Indicates severe injury or unconsciousness

**Implementation:**
```cpp
if (confirmedFall && !immobilityDetected) {
    if (abs(ax) < 1000 && abs(ay) < 1000 && abs(az) < 1000) {
        if (immobilityStartTime == 0) {
            immobilityStartTime = millis();
        }
        if (millis() - immobilityStartTime > 5000) {
            Serial.println("🚨 POSSIBLE UNCONSCIOUSNESS!");
            immobilityDetected = true;
        }
    } else {
        immobilityStartTime = 0;  // Reset if movement detected
    }
}
```

**Rationale:** Prolonged immobility after a fall suggests serious injury requiring immediate medical attention.

### State Machine Diagram

```
[Idle] ──Acc<0.5g──► [Free Fall]
                       │
                       │ 500ms window
                       ▼
              [Impact Detected] ◄──Acc>2.5g──┐
                       │                      │
                       │ 2s tilt >80°         │
                       ▼                      │
              [Fall Confirmed]                │
                       │                      │
                       │ 5s immobility        │
                       ▼                      │
              [Immobility Alert]              │
                       │                      │
                       │ 30s timeout          │
                       └──────────────────────┘
                              [Reset]
```

### Threshold Calibration

Default thresholds are suitable for most adults but can be adjusted:

| Parameter | Default | Range | Adjustment |
|-----------|---------|-------|------------|
| Free Fall Threshold | 0.5g | 0.3-0.7g | Lower = more sensitive |
| Impact Threshold | 2.5g | 2.0-4.0g | Lower = more sensitive |
| Tilt Angle | 80° | 60-90° | Lower = more sensitive |
| Tilt Duration | 2000ms | 1000-5000ms | Shorter = faster detection |
| Immobility Duration | 5000ms | 3000-10000ms | Shorter = faster alert |

---

## Machine Learning Models

### Model Architecture

The system employs RandomForest classifiers for both fall detection and stress prediction due to their:

- **Robustness**: Handles noisy sensor data well
- **Interpretability**: Feature importance analysis possible
- **Efficiency**: Fast inference suitable for real-time applications
- **Non-linearity**: Captures complex relationships without explicit feature engineering

### Fall Detection Model

**Model Specifications:**

| Attribute | Value |
|-----------|-------|
| Algorithm | RandomForest Classifier |
| Number of Estimators | 100 |
| Max Depth | None (fully grown) |
| Criterion | Gini impurity |
| Features | 9 |
| Classes | 2 (Fall / No Fall) |

**Feature Engineering:**

The model uses 9 carefully selected features derived from raw IMU data:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| acc_max | Maximum acceleration magnitude | Captures peak impact forces |
| gyro_max | Maximum gyroscope magnitude | Detects rapid rotations |
| acc_kurtosis | Acceleration kurtosis | Identifies impact sharpness |
| gyro_kurtosis | Gyroscope kurtosis | Detects rotational spikes |
| lin_max | Maximum linear acceleration | Isolates translational motion |
| acc_skewness | Acceleration skewness | Indicates asymmetric forces |
| gyro_skewness | Gyroscope skewness | Shows rotational bias |
| post_gyro_max | Post-fall gyroscope max | Detects post-impact motion |
| post_lin_max | Post-fall linear max | Captures sliding/scrambling |

**Training Data:**

The model was trained on labeled fall datasets including:
- Simulated falls (forward, backward, sideways)
- Activities of daily living (walking, sitting, bending)
- Near-falls and recovery motions

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Accuracy | >95% |
| Sensitivity | >98% (minimizes missed falls) |
| Specificity | >92% (minimizes false alarms) |
| Inference Time | <10ms |

### Stress Prediction Model

**Model Specifications:**

| Attribute | Value |
|-----------|-------|
| Algorithm | RandomForest Classifier |
| Number of Estimators | 100 |
| Features | 27+ HRV metrics |
| Classes | 3 (No Stress, Interruption, Time Pressure) |

**Feature Categories:**

1. **Time Domain Features (9)**
   - MEAN_RR, MEDIAN_RR, HR
   - SDRR, RMSSD, SDSD
   - SDRR_RMSSD ratio
   - pNN25, pNN50

2. **Frequency Domain Features (12)**
   - VLF, LF, HF (raw power)
   - VLF_PCT, LF_PCT, HF_PCT (percentages)
   - LF_NU, HF_NU (normalized units)
   - TP (total power)
   - LF_HF, HF_LF (ratios)

3. **Nonlinear Features (6+)**
   - SD1, SD2 (Poincaré plot)
   - Sample Entropy
   - Higuchi Fractal Dimension
   - KURT, SKEW

**Stress Classification:**

| Class | Description | HRV Pattern |
|-------|-------------|-------------|
| 0 - No Stress | Relaxed state | High RMSSD, high HF |
| 1 - Interruption | External disturbance | Reduced HRV, increased LF |
| 2 - Time Pressure | Mental workload | Very low HRV, high LF/HF |

**Training Pipeline:**

```python
# Data Sources
time_df = pd.read_csv("time_domain_features_train.csv")
nonlinear_df = pd.read_csv("heart_rate_non_linear_features_train.csv")
frequency_df = pd.read_csv("frequency_domain_features_train.csv")

# Label Mapping
label_map = {
    'no stress': 0,
    'interruption': 1,
    'time pressure': 2
}

# Preprocessing
1. Merge all feature dataframes
2. Extract labels from 'condition' column
3. Align rows across datasets
4. Handle missing values (mean imputation)
5. Scale features (StandardScaler)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save artifacts
joblib.dump(model, "stress_model.pkl")
joblib.dump(scaler, "stress_scaler.pkl")
```

---

## Backend API

### Flask Server Architecture

The Flask server (`app.py`) serves as the central hub for data processing, ML inference, and real-time communication.

**Core Components:**

1. **REST API Layer**: HTTP endpoints for data reception and predictions
2. **ML Inference Engine**: Loads and executes trained models
3. **WebSocket Manager**: Real-time bidirectional communication
4. **Data Pipeline**: Processing and validation of incoming sensor data

### Dependencies

```
flask>=2.0.0          # Web framework
flask-socketio>=5.0.0 # WebSocket support
flask-cors>=3.0.0     # Cross-origin requests
scikit-learn>=1.0.0   # Machine learning
joblib>=1.0.0         # Model serialization
numpy>=1.20.0         # Numerical operations
```

### Data Processing Pipeline

```
ESP32 Data → HTTP POST → Key Mapping → Feature Extraction
                                              ↓
                                    ML Model Inference
                                              ↓
                              WebSocket Emit → Dashboard
```

**Key Mapping:**

The server maps incoming JSON keys to expected ML feature names:

```python
key_map = {
    "meanRR": "MEAN_RR",
    "medianRR": "MEDIAN_RR",
    "hr": "HR",
    "rmssd": "RMSSD",
    "sdr": "SDRR",
    # ... 30+ mappings
}
```

### Auto-Prediction Feature

The server automatically performs stress prediction when HRV data is received:

```python
if stress_model is not None and stress_scaler is not None:
    features = np.array([[float(mapped_data[k]) for k in required_keys]])
    features_scaled = stress_scaler.transform(features)
    prediction = stress_model.predict(features_scaled)[0]
    
    # Map to human-readable labels
    label_map = {0: "no stress", 1: "interruption", 2: "time pressure"}
    result = label_map.get(prediction, "Unknown")
    
    # Emit to dashboard
    socketio.emit('stress_prediction', stress_result)
```

---

## Web Dashboard

### Dashboard Architecture

The web dashboard (`index.html`) provides a comprehensive real-time visualization interface built with modern web technologies.

**Technology Stack:**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | Bootstrap 5.3 | Responsive layout |
| Charts | Plotly.js 2.24.2 | Interactive visualizations |
| Real-time | Socket.IO 4.6.1 | WebSocket communication |
| DOM Manipulation | jQuery 3.6.0 | Dynamic content updates |
| Time Formatting | Moment.js 2.29.4 | Timestamp display |

### Dashboard Sections

#### 1. Connection Panel
- Server IP and port configuration
- Connection status indicator (connected/disconnected)
- Connect/Disconnect controls
- Visual feedback for connection state

#### 2. Vital Signs Display
Real-time display of four key metrics with color-coded alerts:

| Metric | Normal Range | Warning | Danger |
|--------|--------------|---------|--------|
| Heart Rate | 60-100 BPM | <60 or >100 | <50 or >120 |
| SpO2 | 95-100% | 90-94% | <90% |
| Blood Pressure | 90-120/60-80 | 120-139/80-89 | >140/90 |
| Temperature | 36-37.5°C | 37.5-38°C | >38°C |

#### 3. Real-Time Charts

**ECG Waveform:**
- Scrolling time-series display
- 100-point buffer for smooth visualization
- Updates with each new ECG sample

**EMG Signal:**
- Similar scrolling display
- Shows muscle activation patterns

**Heart Rate Trend:**
- Time-series plot with 20-point history
- Shows HR variability over time

**SpO2 & Temperature:**
- Dual-axis plot showing both metrics
- Correlation visualization

#### 4. Motion Data

**Accelerometer:**
- X, Y, Z axis progress bars (-5g to +5g)
- Real-time percentage display
- Color-coded bars

**Gyroscope:**
- X, Y, Z axis progress bars (-250°/s to +250°/s)
- Rotation visualization

#### 5. Fall Detection Status

- Visual indicator (green = safe, red = fall detected)
- Alert notifications with auto-dismiss
- Status text display
- Historical fall log

#### 6. Immobility Detection

- Post-fall monitoring display
- Warning indicators
- Timer showing immobility duration

#### 7. Stress Level Gauge

- Circular gauge (0-100 scale)
- Color zones: Green (0-30), Yellow (30-70), Red (70-100)
- Numerical value display
- Stress classification text
- Probability distribution for multi-class output

#### 8. Data Logs

- Rolling log of all incoming data
- Timestamp for each entry
- JSON preview of data packets
- Clear log functionality

### Real-Time Update Mechanism

```javascript
// Socket.IO connection
socket = io(serverUrl);

// Sensor data handler
socket.on('sensor_data', function(data) {
    updateHeartRate(data.hr);
    updateSpO2(data.spo2);
    updateECG(data.ecg);
    updateAccelerometer(data.accelerometer);
    // ... other updates
});

// Stress prediction handler
socket.on('stress_prediction', function(data) {
    updateStressGauge(data.confidence * 100);
    updateStressText(data.prediction);
});
```

---

## Installation Guide

### Prerequisites

**Hardware Requirements:**
- 2x ESP32 development boards
- AD8232 ECG module
- MAX30100 PPG sensor
- MPU6050 IMU module
- MyoWare EMG sensor with cable shield
- 10kΩ NTC thermistor
- Breadboard and jumper wires
- USB cables for programming

**Software Requirements:**
- Arduino IDE (1.8.x or 2.x)
- Python 3.8+
- Web browser (Chrome/Firefox/Edge)

### Step 1: Hardware Assembly

**ECG-PPG Module:**
1. Connect AD8232:
   - OUTPUT → ESP32 GPIO33
   - LO+ → ESP32 GPIO35
   - LO- → ESP32 GPIO32
   - 3.3V → ESP32 3.3V
   - GND → ESP32 GND

2. Connect MAX30100:
   - VIN → ESP32 3.3V
   - GND → ESP32 GND
   - SDA → ESP32 GPIO21
   - SCL → ESP32 GPIO22

**Motion-EMG Module:**
1. Connect MPU6050:
   - VCC → ESP32 3.3V
   - GND → ESP32 GND
   - SDA → ESP32 GPIO21
   - SCL → ESP32 GPIO22

2. Connect MyoWare:
   - SIG → ESP32 GPIO34
   - VCC → ESP32 3.3V
   - GND → ESP32 GND

3. Connect NTC Thermistor:
   - Voltage divider with 10kΩ resistor
   - Midpoint → ESP32 GPIO36
   - Other end → 3.3V and GND

### Step 2: Arduino Setup

1. Install Arduino IDE
2. Add ESP32 board support:
   - File → Preferences → Additional Board Manager URLs
   - Add: `https://dl.espressif.com/dl/package_esp32_index.json`
   - Tools → Board → Board Manager → Search "ESP32" → Install

3. Install required libraries:
   - Wire (built-in)
   - WiFi (built-in)
   - HTTPClient (built-in)
   - ArduinoJson
