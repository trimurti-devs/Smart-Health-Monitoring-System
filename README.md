<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Health Monitoring System</title>
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --success: #4cc9f0;
            --warning: #f72585;
            --danger: #ff006e;
            --dark: #212529;
            --light: #f8f9fa;
            --gradient: linear-gradient(135deg, #4361ee 0%, #4895ef 100%);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            line-height: 1.7;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
        }
        
        .header {
            background: var(--gradient);
            color: white;
            padding: 60px 40px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 40px;
            box-shadow: 0 10px 40px rgba(67, 97, 238, 0.3);
        }
        
        .header h1 { font-size: 3em; margin-bottom: 15px; }
        .header p { font-size: 1.3em; opacity: 0.95; }
        
        .badges { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 25px; }
        .badge { background: rgba(255,255,255,0.2); padding: 8px 20px; border-radius: 50px; font-size: 0.9em; font-weight: 600; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.3); }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; margin: 30px 0; }
        
        .card { background: linear-gradient(145deg, #fff 0%, #f8f9fa 100%); border-radius: 15px; padding: 25px; box-shadow: 0 5px 20px rgba(0,0,0,0.08); border: 1px solid rgba(0,0,0,0.05); transition: all 0.3s ease; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0,0,0,0.12); }
        .card-icon { font-size: 2.5em; margin-bottom: 15px; }
        .card h3 { color: var(--primary); margin-bottom: 10px; font-size: 1.3em; }
        
        .section { margin: 50px 0; }
        .section-title { font-size: 2em; color: var(--dark); margin-bottom: 25px; padding-bottom: 15px; border-bottom: 3px solid var(--primary); display: inline-block; }
        
        .arch-diagram { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; border-radius: 15px; font-family: 'Consolas', monospace; overflow-x: auto; text-align: center; }
        .arch-diagram .box { display: inline-block; background: var(--primary); padding: 15px 25px; border-radius: 10px; margin: 10px; text-align: center; }
        .arch-diagram .arrow { color: var(--accent); font-size: 1.5em; margin: 0 10px; }
        
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 5px 20px rgba(0,0,0,0.08); }
        th { background: var(--gradient); color: white; padding: 15px; text-align: left; font-weight: 600; }
        td { padding: 12px 15px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f8f9fa; }
        
        code { background: #1a1a2e; color: #4cc9f0; padding: 2px 8px; border-radius: 5px; font-family: 'Consolas', monospace; }
        pre { background: #1a1a2e; color: #e0e0e0; padding: 20px; border-radius: 10px; overflow-x: auto; font-size: 0.9em; line-height: 1.5; }
        
        .feature-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
        .feature-item { display: flex; align-items: center; gap: 12px; padding: 15px; background: #f8f9fa; border-radius: 10px; transition: all 0.3s ease; }
        .feature-item:hover { background: #e9ecef; transform: translateX(5px); }
        .feature-icon { font-size: 1.5em; }
        
        .step { display: flex; align-items: flex-start; gap: 20px; margin: 20px 0; padding: 20px; background: linear-gradient(145deg, #fff 0%, #f8f9fa 100%); border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
        .step-number { background: var(--gradient); color: white; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2em; flex-shrink: 0; }
        
        .json-block { background: #1e1e3f; color: #a9b7c6; padding: 20px; border-radius: 10px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 0.85em; line-height: 1.6; }
        .json-key { color: #4cc9f0; }
        .json-string { color: #6a8759; }
        .json-number { color: #b5cea8; }
        
        .alert { padding: 20px; border-radius: 10px; margin: 20px 0; }
        .alert-info { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-left: 5px solid #2196f3; }
        .alert-warning { background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-left: 5px solid #ff9800; }
        
        .footer { text-align: center; padding: 30px; margin-top: 50px; border-top: 2px solid #eee; color: #666; }
        
        h3 { color: var(--dark); margin: 25px 0 15px 0; font-size: 1.4em; }
        h4 { color: var(--secondary); margin: 20px 0 10px 0; font-size: 1.1em; }
        
        hr { border: none; border-top: 1px solid #eee; margin: 20px 0; }
        
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 2em; }
            .section-title { font-size: 1.5em; }
            .arch-diagram { font-size: 0.8em; }
            .two-col { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<div class="header">
    <h1>🫀 Smart Health Monitoring System</h1>
    <p>A comprehensive real-time health monitoring system integrating multiple biometric sensors with machine learning for fall detection and stress prediction.</p>
    <div class="badges">
        <span class="badge">🟢 ESP32</span>
        <span class="badge">🟦 Flask</span>
        <span class="badge">🟠 Machine Learning</span>
        <span class="badge">🟣 SocketIO</span>
        <span class="badge">📊 Real-time</span>
    </div>
</div>

<div class="section">
    <h2 class="section-title">✨ Key Features</h2>
    <div class="feature-list">
        <div class="feature-item"><span class="feature-icon">🫀</span> <div><strong>ECG Monitoring</strong><br><small>Real-time heart rate with R-peak detection</small></div></div>
        <div class="feature-item"><span class="feature-icon">💓</span> <div><strong>SpO2 & BP</strong><br><small>Pulse oximetry and blood pressure estimation</small></div></div>
        <div class="feature-item"><span class="feature-icon">📊</span> <div><strong>30+ HRV Metrics</strong><br><small>Time, frequency, and nonlinear analysis</small></div></div>
        <div class="feature-item"><span class="feature-icon">🤖</span> <div><strong>AI Fall Detection</strong><br><small>4-stage ML-based algorithm</small></div></div>
        <div class="feature-item"><span class="feature-icon">😰</span> <div><strong>Stress Prediction</strong><br><small>Real-time HRV-based assessment</small></div></div>
        <div class="feature-item"><span class="feature-icon">📈</span> <div><strong>Live Dashboard</strong><br><small>Web-based real-time visualization</small></div></div>
        <div class="feature-item"><span class="feature-icon">🔄</span> <div><strong>WebSocket</strong><br><small>Bidirectional real-time communication</small></div></div>
        <div class="feature-item"><span class="feature-icon">📳</span> <div><strong>Motion Tracking</strong><br><small>6-axis IMU with fall detection</small></div></div>
    </div>
</div>

<div class="section">
    <h2 class="section-title">🏗️ System Architecture</h2>
    <div class="arch-diagram">
        <span class="box">ESP32 #1<br><small>ECG/PPG</small></span>
        <span class="arrow">→</span>
        <span class="box">ESP32 #2<br><small>Gyro/EMG</small></span>
        <span class="arrow">→</span>
        <span class="box">Flask Server<br><small>Port 5000</small></span>
        <span class="arrow">→</span>
        <span class="box">Web Dashboard<br><small>SocketIO</small></span>
    </div>
</div>

<div class="section">
    <h2 class="section-title">🖥️ Hardware Components</h2>
    
    <h3>ESP32 Microcontroller</h3>
    <table>
        <tr><th>Component</th><th>Specification</th></tr>
        <tr><td>MCU</td><td>Dual-core Xtensa® 32-bit LX6 @ 240MHz</td></tr>
        <tr><td>WiFi</td><td>802.11 b/g/n</td></tr>
        <tr><td>Bluetooth</td><td>Bluetooth 4.2</td></tr>
        <tr><td>ADC</td><td>12-bit (0-4095 range)</td></tr>
        <tr><td>Interfaces</td><td>I2C, SPI, UART</td></tr>
    </table>
    
    <h3>Sensors</h3>
    <table>
        <tr><th>Sensor</th><th>Model</th><th>Interface</th><th>Purpose</th></tr>
        <tr><td>🫀 ECG</td><td>AD8232</td><td>Analog</td><td>Heart rate & rhythm monitoring</td></tr>
        <tr><td>💓 PPG</td><td>MAX30100</td><td>I2C</td><td>SpO2 & pulse oximetry</td></tr>
        <tr><td>📳 Accelerometer</td><td>MPU6050</td><td>I2C</td><td>Motion & fall detection</td></tr>
        <tr><td>🔄 Gyroscope</td><td>MPU6050</td><td>I2C</td><td>Rotation & orientation</td></tr>
        <tr><td>💪 EMG</td><td>MyoWare Cable Shield</td><td>Analog</td><td>Muscle activity</td></tr>
        <tr><td>🌡️ Temperature</td><td>NTC Thermistor (10kΩ)</td><td>Analog</td><td>Body temperature</td></tr>
    </table>
    
    <h3>Pin Configuration</h3>
    
    <h4>ECG-PPG ESP32 (Device 1)</h4>
    <table>
        <tr><th>Pin</th><th>Function</th><th>Notes</th></tr>
        <tr><td>GPIO33</td><td>ECG Analog Input</td><td>ECG signal from AD8232</td></tr>
        <tr><td>GPIO35</td><td>LO+ (Lead Off Detect)</td><td>Lead-off detection positive</td></tr>
        <tr><td>GPIO32</td><td>LO- (Lead Off Detect)</td><td>Lead-off detection negative</td></tr>
        <tr><td>GPIO21</td><td>I2C SDA (MAX30100)</td><td>I2C data line</td></tr>
        <tr><td>GPIO22</td><td>I2C SCL (MAX30100)</td><td>I2C clock line</td></tr>
    </table>
    
    <h4>Gyro-EMG-Temp ESP32 (Device 2)</h4>
    <table>
        <tr><th>Pin</th><th>Function</th><th>Notes</th></tr>
        <tr><td>GPIO34</td><td>EMG Analog Input</td><td>EMG signal from MyoWare</td></tr>
        <tr><td>GPIO36</td><td>NTC Thermistor (VP)</td><td>Temperature sensor input</td></tr>
        <tr><td>GPIO21</td><td>I2C SDA (MPU6050)</td><td>I2C data line</td></tr>
        <tr><td>GPIO22</td><td>I2C SCL (MPU6050)</td><td>I2C clock line</td></tr>
    </table>
</div>

<div class="section">
    <h2 class="section-title">📟 Firmware</h2>
    
    <div class="grid">
        <div class="card">
            <div class="card-icon">🫀</div>
            <h3>ECG-PPG-Stress.ino</h3>
            <p><strong>ECG Processing:</strong></p>
            <ul>
                <li>R-peak detection algorithm</li>
                <li>Heart rate calculation (BPM)</li>
                <li>Lead-off detection</li>
                <li>Noise filtering</li>
            </ul>
            <p><strong>PPG Processing:</strong></p>
            <ul>
                <li>SpO2 calculation using red/IR ratio</li>
                <li>Beat detection with callback</li>
                <li>Signal quality assessment</li>
                <li>Automatic sensor recovery</li>
            </ul>
            <p><strong>Blood Pressure:</strong></p>
            <ul>
                <li>Pulse Transit Time (PTT) calculation</li>
                <li>Systolic/Diastolic estimation</li>
                <li>Calibration-ready interface</li>
            </ul>
        </div>
        
        <div class="card">
            <div class="card-icon">📳</div>
            <h3>Gyro-EMG-Temp.ino</h3>
            <p><strong>Motion Tracking:</strong></p>
            <ul>
                <li>6-axis IMU data (accel + gyro)</li>
                <li>Real-time orientation calculation</li>
                <li>Movement pattern analysis</li>
            </ul>
            <p><strong>EMG Monitoring:</strong></p>
            <ul>
                <li>Raw EMG signal acquisition</li>
                <li>Muscle activity visualization</li>
                <li>Fatigue analysis ready</li>
            </ul>
            <p><strong>Temperature Sensing:</strong></p>
            <ul>
                <li>NTC thermistor interface</li>
                <li>Steinhart-Hart equation conversion</li>
                <li>Body temperature range (30-42°C)</li>
            </ul>
        </div>
    </div>
    
    <h3>Key Firmware Functions</h3>
    <table>
        <tr><th>Function</th><th>Description</th></tr>
        <tr><td><code>interpolateRR()</code></td><td>Resamples RR intervals to uniform sampling frequency for FFT</td></tr>
        <tr><td><code>calculatePSD()</code></td><td>Computes Power Spectral Density using FFT</td></tr>
        <tr><td><code>bandPower()</code></td><td>Calculates power in VLF, LF, HF frequency bands</td></tr>
        <tr><td><code>calculateSampen()</code></td><td>Sample Entropy - measures complexity of RR interval series</td></tr>
        <tr><td><code>calculateHiguchi()</code></td><td>Higuchi Fractal Dimension - measures self-similarity</td></tr>
        <tr><td><code>calculateSD1()</code></td><td>SD1 from Poincaré plot - short-term HRV variability</td></tr>
        <tr><td><code>calculateSD2()</code></td><td>SD2 from Poincaré plot - long-term HRV variability</td></tr>
    </table>
</div>

<div class="section">
    <h2 class="section-title">📊 Advanced HRV Analysis</h2>
    
    <h3>Time Domain Metrics</h3>
    <table>
        <tr><th>Metric</th><th>Description</th><th>Clinical Significance</th></tr>
        <tr><td>meanRR</td><td>Mean RR interval (ms)</td><td>Average heart rate</td></tr>
        <tr><td>medianRR</td><td>Median RR interval</td><td>Robust heart rate estimate</td></tr>
        <tr><td>SDRR</td><td>Standard deviation of RR</td><td>Overall HRV</td></tr>
        <tr><td>RMSSD</td><td>Root mean square of successive differences</td><td>Parasympathetic activity</td></tr>
        <tr><td>SDSD</td><td>Standard deviation of successive differences</td><td>Short-term variability</td></tr>
        <tr><td>pNN25</td><td>% RR intervals differing >25ms</td><td>Fast HRV component</td></tr>
        <tr><td>pNN50</td><td>% RR intervals differing >50ms</td><td>Parasympathetic marker</td></tr>
        <tr><td>SDRR/RMSSD</td><td>Ratio indicator</td><td>HRV balance</td></tr>
    </table>
    
    <h3>Frequency Domain Metrics</h3>
    <table>
        <tr><th>Band</th><th>Frequency Range</th><th>Associated With</th></tr>
        <tr><td>VLF</td><td>0.003-0.04 Hz</td><td>Thermoregulatory, peripheral vasomotor</td></tr>
        <tr><td>LF</td><td>0.04-0.15 Hz</td><td>Sympathetic & parasympathetic</td></tr>
        <tr><td>HF</td><td>0.15-0.4 Hz</td><td>Parasympathetic (vagal)</td></tr>
    </table>
    <p><em>Additional: LF/HF ratio, LF nu, HF nu, VLF%, LF%, HF%, Total Power</em></p>
    
    <h3>Nonlinear Metrics</h3>
    <table>
        <tr><th>Metric</th><th>Description</th></tr>
        <tr><td>SD1</td><td>Poincaré plot short-axis (parasympathetic)</td></tr>
        <tr><td>SD2</td><td>Poincaré plot long-axis (overall HRV)</td></tr>
        <tr><td>Sample Entropy</td><td>Signal complexity measure</td></tr>
        <tr><td>Higuchi FD</td><td>Fractal dimension</td></tr>
        <tr><td>Kurtosis</td><td>Distribution tail weight</td></tr>
        <tr><td>Skewness</td><td>Distribution asymmetry</td></tr>
    </table>
</div>

<div class="section">
    <h2 class="section-title">⚠️ Fall Detection Algorithm</h2>
    
    <p>The 4-stage fall detection system implements a comprehensive fall classification approach:</p>
    
    <table>
        <tr><th>Stage</th><th>Trigger</th><th>Threshold</th><th>Duration</th></tr>
        <tr><td>1. Free Fall</td><td>Acc < 0.5g</td><td>< 4.9 m/s²</td><td>Instant</td></tr>
        <tr><td>2. Impact</td><td>Acc > 2.5g</td><td>> 24.5 m/s²</td><td>500ms window</td></tr>
        <tr><td>3. Tilt</td><td>Angle > 80°</td><td>> 80 degrees</td><td>2+ seconds</td></tr>
        <tr><td>4. Immobility</td><td>No movement</td><td>All axes < 1000</td><td>5+ seconds</td></tr>
    </table>
    
    <div class="alert alert-info">
        <strong>Algorithm Flow:</strong> Continuous Monitoring → Free Fall Detection → Impact Detection → Tilt Detection → Fall Confirmed → Immobility Check → Alert
    </div>
</div>

<div class="section">
    <h2 class="section-title">🧠 Machine Learning Models</h2>
    
    <div class="two-col">
        <div class="card">
            <h3>🤖 Fall Detection</h3>
            <p><strong>Algorithm:</strong> RandomForest Classifier</p>
            <p><strong>Type:</strong> Binary Classification</p>
            <p><strong>Features (9):</strong></p>
            <ul>
                <li>acc_max, gyro_max</li>
                <li>acc_kurtosis, gyro_kurtosis</li>
                <li>lin_max</li>
                <li>acc_skewness, gyro_skewness</li>
                <li>post_gyro_max, post_lin_max</li>
            </ul>
            <p><strong>Output:</strong> Fall (1) / No Fall (0)</p>
        </div>
        
        <div class="card">
            <h3>😰 Stress Prediction</h3>
            <p><strong>Algorithm:</strong> RandomForest Classifier</p>
            <p><strong>Type:</strong> Multi-class Classification</p>
            <p><strong>Features (27+):</strong></p>
            <ul>
                <li>Time domain: MEAN_RR, SDRR, RMSSD, SDSD, etc.</li>
                <li>Frequency: VLF, LF, HF, LF/HF</li>
                <li>Nonlinear: SD1, SD2, Sample Entropy, Higuchi</li>
            </ul>
            <p><strong>Classes:</strong> No Stress, Interruption, Time Pressure</p>
        </div>
    </div>
    
    <h3>Model Training Pipeline (model.py)</h3>
    <pre># Data Sources:
# - time_domain_features_train.csv
# - heart_rate_non_linear_features_train.csv
# - frequency_domain_features_train.csv

# Processing Steps:
# 1. Load and merge all feature CSV files
# 2. Extract labels from 'condition' column
# 3. Align rows across datasets
# 4. Handle missing values (mean imputation)
# 5. Scale features with StandardScaler
# 6. Train RandomForest (n_estimators=10)
# 7. Save model and scaler as .pkl files</pre>
</div>

<div class="section">
    <h2 class="section-title">🚀 Installation & Setup</h2>
    
    <div class="step">
        <div class="step-number">1</div>
        <div>
            <h3>Hardware Setup</h3>
            <p>Connect sensors according to pin configuration. Ensure:</p>
            <ul>
                <li>3.3V power supply for all sensors</li>
                <li>Proper grounding (common GND)</li>
                <li>ECG electrodes properly placed on body</li>
                <li>MPU6050 mounted on body/wrist</li>
            </ul>
        </div>
    </div>
    
    <div class="step">
        <div class="step-number">2</div>
        <div>
            <h3>WiFi Configuration</h3>
            <p>Edit in both .ino files:</p>
            <pre>WiFi.begin("Your_SSID", "Your_Password");</pre>
        </div>
    </div>
    
    <div class="step">
        <div class="step-number">3</div>
        <div>
            <h3>Server IP Configuration</h3>
            <pre>const char* serverUrl = "http://192.168.1.8:5000/receive_data";</pre>
        </div>
    </div>
    
    <div class="step">
        <div class="step-number">4</div>
        <div>
            <h3>Python Environment</h3>
            <pre># Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install flask flask-socketio flask-cors scikit-learn joblib numpy</pre>
        </div>
    </div>
    
    <div class="step">
        <div class="step-number">5</div>
        <div>
            <h3>Run the Server</h3>
            <pre>cd "VS Code"
python app.py</pre>
            <p>Server starts at: <code>http://0.0.0.0:5000</code></p>
        </div>
    </div>
    
    <div class="step">
        <div class="step-number">6</div>
        <div>
            <h3>Access Dashboard</h3>
            <p>Open <code>VS Code/index.html</code> in web browser. Enter server IP (e.g., <code>192.168.1.8</code>) and port <code>5000</code>.</p>
        </div>
    </div>
</div>

<div class="section">
    <h2 class="section-title">📡 API Endpoints</h2>
    <table>
        <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
        <tr><td>GET</td><td>/</td><td>Server status</td></tr>
        <tr><td>GET</td><td>/version</td><td>API version info</td></tr>
        <tr><td>POST</td><td>/receive_data</td><td>Receive sensor data from ESP32</td></tr>
        <tr><td>POST</td><td>/predict_fall</td><td>Fall detection prediction</td></tr>
        <tr><td>POST</td><td>/predict_stress</td><td>Stress level prediction</td></tr>
        <tr><td>POST</td><td>/predict_immobility</td><td>Immobility detection</td></tr>
    </table>
    
    <h3>WebSocket Events</h3>
    <div class="feature-list">
        <div class="feature-item"><span class="feature-icon">📨</span> <strong>sensor_data</strong> - Real-time sensor updates</div>
        <div class="feature-item"><span class="feature-icon">🤖</span> <strong>stress_prediction</strong> - ML stress prediction result</div>
        <div class="feature-item"><span class="feature-icon">⚠️</span> <strong>fall_prediction</strong> - ML fall detection result</div>
        <div class="feature-item"><span class="feature-icon">😴</span> <strong>immobility_prediction</strong> - Immobility detection result</div>
    </div>
</div>

<div class="section">
    <h2 class="section-title">📄 Data Format</h2>
    
    <h3>ECG-PPG Device Data</h3>
    <div class="json-block">
{
  <span class="json-key">"uuid"</span>: <span class="json-string">"patient_uuid_123"</span>,
  <span class="json-key">"datasetId"</span>: <span class="json-string">"dataset_id_001"</span>,
  <span class="json-key">"condition"</span>: <span class="json-string">"healthy"</span>,
  <span class="json-key">"hr"</span>: <span class="json-number">72.5</span>,
  <span class="json-key">"spo2"</span>: <span class="json-number">98.2</span>,
  <span class="json-key">"sys"</span>: <span class="json-number">118.5</span>,
  <span class="json-key">"dia"</span>: <span class="json-number">78.3</span>,
  <span class="json-key">"meanRR"</span>: <span class="json-number">833.5</span>,
  <span class="json-key">"medianRR"</span>: <span class="json-number">830.0</span>,
  <span class="json-key">"sdr"</span>: <span class="json-number">52.1</span>,
  <span class="json-key">"rmssd"</span>: <span class="json-number">45.2</span>,
  <span class="json-key">"sdsd"</span>: <span class="json-number">42.1</span>,
  <span class="json-key">"sdr_rmssd"</span>: <span class="json-number">1.15</span>,
  <span class="json-key">"pnn25"</span>: <span class="json-number">28.5</span>,
  <span class="json-key">"pnn50"</span>: <span class="json-number">15.3</span>,
  <span class="json-key">"kurt"</span>: <span class="json-number">0.85</span>,
  <span class="json-key">"skew"</span>: <span class="json-number">0.12</span>,
  <span class="json-key">"meanRelRR"</span>: <span class="json-number">0.02</span>,
  <span class="json-key">"vlf"</span>: <span class="json-number">120.5</span>,
  <span class="json-key">"vlf_pct"</span>: <span class="json-number">12.5</span>,
  <span class="json-key">"lf"</span>: <span class="json-number">340.2</span>,
  <span class="json-key">"lf_pct"</span>: <span class="json-number">35.2</span>,
  <span class="json-key">"lf_nu"</span>: <span class="json-number">45.8</span>,
  <span class="json-key">"hf"</span>: <span class="json-number">280.1</span>,
  <span class="json-key">"hf_pct"</span>: <span class="json-number">29.0</span>,
  <span class="json-key">"hf_nu"</span>: <span class="json-number">37.7</span>,
  <span class="json-key">"tp"</span>: <span class="json-number">965.3</span>,
  <span class="json-key">"lf_hf"</span>: <span class="json-number">1.21</span>,
  <span class="json-key">"hf_lf"</span>: <span class="json-number">0.83</span>,
  <span class="json-key">"sd1"</span>: <span class="json-number">32.0</span>,
  <span class="json-key">"sd2"</span>: <span class="json-number">65.5</span>,
  <span class="json-key">"sampen"</span>: <span class="json-number">1.85</span>,
  <span class="json-key">"higuchi"</span>: <span class="json-number">1.92</span>,
  <span class="json-key">"ecg"</span>: <span class="json-number">2048</span>
}</div>
    
    <h3>Gyro-EMG-Temp Device Data</h3>
    <div class="json-block">
{
  <span class="json-key">"device_id"</span>: <span class="json-string">"ESP32_1"</span>,
  <span class="json-key">"timestamp"</span>: <span class="json-number">1234567890</span>,
  <span class="json-key">"emg"</span>: <span class="json-number">512</span>,
  <span class="json-key">"temperature"</span>: <span class="json-number">36.5</span>,
  <span class="json-key">"x"</span>: <span class="json-number">0.02</span>,
  <span class="json-key">"y"</span>: <span class="json-number">0.98</span>,
  <span class="json-key">"z"</span>: <span class="json-number">0.15</span>,
  <span class="json-key">"accelerometer"</span>: {
    <span class="json-key">"x"</span>: <span class="json-number">0.02</span>,
    <span class="json-key">"y"</span>: <span class="json-number">0.98</span>,
    <span class="json-key">"z"</span>: <span class="json-number">0.15</span>
  },
  <span class="json-key">"gyroscope"</span>: {
    <span class="json-key">"x"</span>: <span class="json-number">1.2</span>,
    <span class="json-key">"y"</span>: <span class="json-number">-0.5</span>,
    <span class="json-key">"z"</span>: <span class="json-number">0.8</span>
  },
  <span class="json-key">"acc_max"</span>: <span class="json-number">1.02</span>,
  <span class="json-key">"gyro_max"</span>: <span class="json-number">15.3</span>,
  <span class="json-key">"lin_max"</span>: <span class="json-number">0.98</span>,
  <span class="json-key">"acc_skewness"</span>: <span class="json-number">0.12</span>,
  <span class="json-key">"gyro_skewness"</span>: <span class="json-number">0.05</span>,
  <span class="json-key">"fall_detected"</span>: <span class="json-number">0</span>,
  <span class="json-key">"immobility_detected"</span>: <span class="json-number">0</span>
}</div>
</div>

<div class="section">
    <h2 class="section-title">⚙️ Technical Specifications</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Protocol</td><td>HTTP POST / WebSocket</td></tr>
        <tr><td>Data Format</td><td>JSON</td></tr>
        <tr><td>Baud Rate (Serial)</td><td>115200</td></tr>
        <tr><td>Server Port</td><td>5000</td></tr>
        <tr><td>ECG Sampling</td><td>~250 Hz</td></tr>
        <tr><td>PPG Sampling</td><td>50 Hz (20ms interval)</td></tr>
        <tr><td>MPU6050 Sampling</td><td>~100 Hz</td></tr>
        <tr><td>EMG Sampling</td><td>~250 Hz</td></tr>
        <tr><td>Temperature Sampling</td><td>1 Hz</td></tr>
    </table>
    
    <h3>ML Model Specifications</h3>
    <table>
        <tr><th>Model</th><th>Algorithm</th><th>Features</th><th>Classes</th></tr>
        <tr><td>Fall Detection</td><td>RandomForest</td><td>9</td><td>2</td></tr>
        <tr><td>Stress Prediction</td><td>RandomForest</td><td>27+</td><td>3</td></tr>
    </table>
</div>

<div class="section">
    <h2 class="section-title">🔧 Troubleshooting</h2>
    <div class="grid">
        <div class="card">
            <h3>WiFi Connection Failed</h3>
            <ul>
                <li>Verify SSID and password</li>
                <li>Ensure ESP32 within WiFi range</li>
                <li>Confirm 2.4GHz network</li>
            </ul>
        </div>
        <div class="card">
            <h3>Sensor Not Responding</h3>
            <ul>
                <li>Check wiring connections</li>
                <li>Verify 3.3V power supply</li>
                <li>Use Serial Monitor (115200)</li>
            </ul>
        </div>
        <div class="card">
            <h3>Inaccurate HRV Metrics</h3>
            <ul>
                <li>Proper ECG electrode placement</li>
                <li>Minimize motion artifacts</li>
                <li>Verify finger on PPG sensor</li>
            </ul>
        </div>
        <div class="card">
            <h3>False Fall Positives</h3>
            <ul>
                <li>Adjust thresholds in code</li>
                <li>Calibrate sensitivity</li>
                <li>Check accelerometer orientation</li>
            </ul>
        </div>
    </div>
    
    <div class="alert alert-warning">
        <strong>Debug Mode:</strong> Enable debug output in Arduino IDE - Tools → Serial Monitor → 115200 baud
    </div>
</div>

<div class="section">
    <h2 class="section-title">📊 Dashboard Features</h2>
    <div class="feature-list">
        <div class="feature-item"><span class="feature-icon">❤️</span> <strong>Vital Signs</strong> - Heart Rate, SpO2, BP, Temperature</div>
        <div class="feature-item"><span class="feature-icon">📈</span> <strong>Real-time Charts</strong> - ECG, EMG, HR Trend</div>
        <div class="feature-item"><span class="feature-icon">📳</span> <strong>Motion Data</strong> - Accelerometer/Gyroscope</div>
        <div class="feature-item"><span class="feature-icon">⚠️</span> <strong>Fall Detection</strong> - Alert notifications</div>
        <div class="feature-item"><span class="feature-icon">😴</span> <strong>Immobility</strong> - Post-fall monitoring</div>
        <div class="feature-item"><span class="feature-icon">😰</span> <strong>Stress Level</strong> - Gauge visualization</div>
        <div class="feature-item"><span class="feature-icon">📝</span> <strong>Data Logs</strong> - Rolling log display</div>
    </div>
</div>

<div class="section">
    <h2 class="section-title">📝 Backend API (app.py)</h2>
    
    <h3>Dependencies</h3>
    <pre>flask>=2.0.0
flask-socketio>=5.0.0
flask-cors>=3.0.0
scikit-learn>=1.0.0
joblib>=1.0.0
numpy>=1.20.0</pre>
    
    <h3>Features</h3>
    <ul>
        <li>RESTful API endpoints</li>
        <li>Real-time WebSocket updates via SocketIO</li>
        <li>Automatic stress prediction from HRV data</li>
        <li>ML model inference for fall and stress detection</li>
        <li>Automatic model retraining capability</li>
        <li>CORS support for cross-origin requests</li>
    </ul>
</div>

<div class="alert alert-info">
    <strong>📌 Note:</strong> This project is for educational and research purposes.
</div>

<div class="footer">
    <p><strong>Version:</strong> 2.1.0 | <strong>API:</strong> Flask + SocketIO</p>
    <p>🫀 Smart Health Monitoring System</p>
</div>

</body>
</html>
