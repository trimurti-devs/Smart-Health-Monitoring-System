#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "MAX30100_PulseOximeter.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <arduinoFFT.h>
#include <math.h>

#define FFT_SIZE 128
float vReal[FFT_SIZE];
float vImag[FFT_SIZE];
ArduinoFFT<float> fft;

float samplingFrequency = 4.0; // Hz, typical resampling frequency for HRV spectral analysis

#define ECG_PIN     33
#define LO_PLUS     35
#define LO_MINUS    32

PulseOximeter pox;

unsigned long currentMillis = 0;
unsigned long lastPoxUpdate = 0;
const int poxUpdateInterval = 20;

unsigned long rPeakTime = 0;
unsigned long ppgPeakTime = 0;
unsigned long tsLastReport = 0;
unsigned long lastGoodRead = 0;

bool newRPeak = false;
bool newPPGPeak = false;
bool leadOff = false;

float heartRate = 0;
float SpO2 = 0;
std::vector<unsigned long> rrIntervals;  // Store RR intervals timestamps

// Server URL
const char* serverUrl = "http://192.168.1.8:5000/receive_data"; // <-- Set your Flask IP here

// Patient data
String uuid = "patient_uuid_123";  // Unique patient ID
String datasetId = "dataset_id_001";  // Dataset ID
String condition = "healthy";  // Patient's condition

// Beat detection callback
void onBeatDetected() {
  ppgPeakTime = millis();
  newPPGPeak = true;
  lastGoodRead = millis();
}

// Finger presence detection
bool isFingerDetected(float spo2, float hr) {
  return (spo2 > 70 && spo2 <= 100 && hr > 30 && hr < 180);
}

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  WiFi.begin("Argha", "Argha123");  // <---change your WiFi SSID and Password
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  if (!pox.begin()) {
    Serial.println("MAX30100 init failed. Halting.");
    while (1);
  }

  pox.setIRLedCurrent(MAX30100_LED_CURR_11MA);
  pox.setOnBeatDetectedCallback(onBeatDetected);
}

// Helper functions declarations (same as before, omitted here for brevity)
// ...


// Interpolate RR intervals to evenly spaced time series for FFT
void interpolateRR(const std::vector<long>& rrDiffs, std::vector<double>& interpolated, double fs) {
  // Linear interpolation to fs Hz sampling frequency
  if (rrDiffs.size() < 2) return;

  double totalTime = 0;
  for (size_t i = 0; i < rrDiffs.size(); ++i) {
    totalTime += rrDiffs[i] / 1000.0; // convert ms to seconds
  }
  int nSamples = (int)(totalTime * fs);
  if (nSamples < 2) return;

  interpolated.clear();
  interpolated.resize(nSamples, 0);

  // Create time vector for RR intervals
  std::vector<double> timeRR(rrDiffs.size());
  timeRR[0] = 0;
  for (size_t i = 1; i < rrDiffs.size(); ++i) {
    timeRR[i] = timeRR[i - 1] + rrDiffs[i] / 1000.0;
  }

  // Interpolate linearly
  double dt = 1.0 / fs;
  for (int i = 0; i < nSamples; ++i) {
    double t = i * dt;
    // Find interval
    size_t idx = 0;
    while (idx < timeRR.size() - 1 && timeRR[idx + 1] < t) {
      idx++;
    }
    if (idx >= timeRR.size() - 1) {
      interpolated[i] = rrDiffs.back() / 1000.0; // last value in seconds
    } else {
      double t1 = timeRR[idx];
      double t2 = timeRR[idx + 1];
      double v1 = rrDiffs[idx] / 1000.0;
      double v2 = rrDiffs[idx + 1] / 1000.0;
      double val = v1 + (v2 - v1) * (t - t1) / (t2 - t1);
      interpolated[i] = val;
    }
  }
}

// Calculate power spectral density using FFT on interpolated RR intervals
void calculatePSD(const std::vector<double>& interpolated, std::vector<double>& psd) {
  int N = interpolated.size();
  if (N > FFT_SIZE) N = FFT_SIZE;

  for (int i = 0; i < N; i++) {
    vReal[i] = interpolated[i];
    vImag[i] = 0;
  }
  for (int i = N; i < FFT_SIZE; i++) {
    vReal[i] = 0;
    vImag[i] = 0;
  }

  fft.windowing(vReal, FFT_SIZE, FFTWindow::Hamming, FFTDirection::Forward);
  fft.compute(vReal, vImag, FFT_SIZE, FFTDirection::Forward);
  fft.complexToMagnitude(vReal, vImag, FFT_SIZE);

  psd.clear();
  for (int i = 0; i < FFT_SIZE / 2; i++) {
    // Power = magnitude squared normalized by FFT size
    double power = (vReal[i] * vReal[i]) / FFT_SIZE;
    psd.push_back(power);
  }
}

// Calculate band power from PSD
double bandPower(const std::vector<double>& psd, double fs, double lowFreq, double highFreq) {
  double freqResolution = fs / FFT_SIZE;
  double powerSum = 0;
  for (int i = 0; i < psd.size(); i++) {
    double freq = i * freqResolution;
    if (freq >= lowFreq && freq <= highFreq) {
      powerSum += psd[i];
    }
  }
  return powerSum;
}

void loop() {
  currentMillis = millis();

  // Update PPG
  if (currentMillis - lastPoxUpdate >= poxUpdateInterval) {
    pox.update();
    lastPoxUpdate = currentMillis;
  }

  // ECG Reading
  int ecgValue = analogRead(ECG_PIN);
  leadOff = (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1);

  // Simple R-peak detection
  if (ecgValue > 1000 && currentMillis - rPeakTime > 300) {
    rPeakTime = currentMillis;
    newRPeak = true;
    rrIntervals.push_back(currentMillis);  // Save the timestamp of R-peaks
  }

  heartRate = pox.getHeartRate();
  SpO2 = pox.getSpO2();

  // Sensor freeze recovery
  if (currentMillis - lastGoodRead > 2000) {
    Serial.println("MAX30100 not updating. Reinitializing...");
    if (pox.begin()) {
      pox.setIRLedCurrent(MAX30100_LED_CURR_24MA);
      pox.setOnBeatDetectedCallback(onBeatDetected);
    }
    lastGoodRead = currentMillis;
  }

  // Only calculate PTT when new peaks from both ECG and PPG exist
  if (newPPGPeak && newRPeak) {
    float PTT = (ppgPeakTime > rPeakTime) ? (ppgPeakTime - rPeakTime) / 1000.0 : 0;
    // Simple BP estimation (replace with your own calibrated model)
    float sysBP = 120 - (PTT * 50);
    float diaBP = 80 - (PTT * 30);

    newPPGPeak = false;
    newRPeak = false;

    if ((currentMillis - tsLastReport > 1000) && !leadOff && isFingerDetected(SpO2, heartRate)) {
      tsLastReport = currentMillis;

      // Calculate RR intervals in ms from timestamps
      std::vector<long> rrDiffs;
      for (size_t i = 1; i < rrIntervals.size(); ++i) {
        rrDiffs.push_back(rrIntervals[i] - rrIntervals[i - 1]);
      }
      if (rrDiffs.size() < 2) {
        // Not enough data to calculate HRV metrics
        return;
      }

      // Calculate mean and median RR intervals
      float meanRR = calculateMean(rrDiffs);
      float medianRR = calculateMedian(rrDiffs);
      float sdr = calculateSDRR(rrDiffs);
      float rmssd = calculateRMSSD(rrDiffs);
      float sdsd = calculateSDSD(rrDiffs);
      float sdr_rmssd = (rmssd != 0) ? (sdr / rmssd) : 0;
      float pnn25 = calculatePNN25(rrDiffs);
      float pnn50 = calculatePNN50(rrDiffs);
      float kurt = calculateKurtosis(rrDiffs);
      float skew = calculateSkewness(rrDiffs);

      // Relative RR intervals (differences normalized by previous RR)
      std::vector<float> relRRIntervals;
      for (size_t i = 1; i < rrDiffs.size(); ++i) {
        relRRIntervals.push_back((float)(rrDiffs[i] - rrDiffs[i - 1]) / rrDiffs[i - 1]);
      }
      if (relRRIntervals.size() < 2) {
        // Not enough data for relative RR metrics
        return;
      }
      float meanRelRR = calculateMeanFloat(relRRIntervals);
      float medianRelRR = calculateMedianFloat(relRRIntervals);
      float sdrRelRR = calculateSDRRFloat(relRRIntervals);
      float rmssdRelRR = calculateRMSSDFloat(relRRIntervals);
      float sdsdRelRR = calculateSDSDFloat(relRRIntervals);
      float sdrRmssdRelRR = (rmssdRelRR != 0) ? (sdrRelRR / rmssdRelRR) : 0;
      float kurtRelRR = calculateKurtosisFloat(relRRIntervals);
      float skewRelRR = calculateSkewnessFloat(relRRIntervals);

      // Interpolate RR intervals for spectral analysis
      std::vector<double> interpolatedRR;
      interpolateRR(rrDiffs, interpolatedRR, samplingFrequency);

      // Calculate power spectral density
      std::vector<double> psd;
      calculatePSD(interpolatedRR, psd);

      // Calculate frequency domain powers
      double vlf_power = bandPower(psd, samplingFrequency, 0.003, 0.04);
      double lf_power = bandPower(psd, samplingFrequency, 0.04, 0.15);
      double hf_power = bandPower(psd, samplingFrequency, 0.15, 0.4);
      double total_power = vlf_power + lf_power + hf_power;

      // Calculate normalized units and percentages
      double lf_nu = (total_power - vlf_power) > 0 ? (lf_power / (total_power - vlf_power)) * 100.0 : 0;
      double hf_nu = (total_power - vlf_power) > 0 ? (hf_power / (total_power - vlf_power)) * 100.0 : 0;
      double vlf_pct = total_power > 0 ? (vlf_power / total_power) * 100.0 : 0;
      double lf_pct = total_power > 0 ? (lf_power / total_power) * 100.0 : 0;
      double hf_pct = total_power > 0 ? (hf_power / total_power) * 100.0 : 0;

      // Calculate LF/HF ratio
      double lf_hf = hf_power > 0 ? lf_power / hf_power : 0;
      double hf_lf = lf_power > 0 ? hf_power / lf_power : 0;

      // Calculate nonlinear metrics
      float sd1 = calculateSD1(rrDiffs);
      float sd2 = calculateSD2(rrDiffs);
      float sampen = calculateSampen(rrDiffs);
      float higuchi = calculateHiguchi(rrDiffs);

      // Display parameters
      Serial.print("Mean RR: "); Serial.println(meanRR);
      Serial.print("Median RR: "); Serial.println(medianRR);
      Serial.print("SDRR: "); Serial.println(sdr);
      Serial.print("RMSSD: "); Serial.println(rmssd);
      Serial.print("SDSD: "); Serial.println(sdsd);
      Serial.print("SDRR/RMSSD: "); Serial.println(sdr_rmssd);
      Serial.print("pNN25: "); Serial.println(pnn25);
      Serial.print("pNN50: "); Serial.println(pnn50);
      Serial.print("Kurtosis: "); Serial.println(kurt);
      Serial.print("Skew: "); Serial.println(skew);
      Serial.print("Mean Rel RR: "); Serial.println(meanRelRR);
      Serial.print("Median Rel RR: "); Serial.println(medianRelRR);
      Serial.print("SDRR Rel RR: "); Serial.println(sdrRelRR);
      Serial.print("RMSSD Rel RR: "); Serial.println(rmssdRelRR);
      Serial.print("SDSD Rel RR: "); Serial.println(sdsdRelRR);
      Serial.print("SDRR/RMSSD Rel RR: "); Serial.println(sdrRmssdRelRR);
      Serial.print("Kurtosis Rel RR: "); Serial.println(kurtRelRR);
      Serial.print("Skewness Rel RR: "); Serial.println(skewRelRR);
      Serial.print("VLF Power: "); Serial.println(vlf_power);
      Serial.print("VLF %: "); Serial.println(vlf_pct);
      Serial.print("LF Power: "); Serial.println(lf_power);
      Serial.print("LF %: "); Serial.println(lf_pct);
      Serial.print("LF nu: "); Serial.println(lf_nu);
      Serial.print("HF Power: "); Serial.println(hf_power);
      Serial.print("HF %: "); Serial.println(hf_pct);
      Serial.print("HF nu: "); Serial.println(hf_nu);
      Serial.print("Total Power: "); Serial.println(total_power);
      Serial.print("LF/HF: "); Serial.println(lf_hf);
      Serial.print("HF/LF: "); Serial.println(hf_lf);
      Serial.print("SD1: "); Serial.println(sd1);
      Serial.print("SD2: "); Serial.println(sd2);
      Serial.print("Sample Entropy: "); Serial.println(sampen);
      Serial.print("Higuchi Fractal Dimension: "); Serial.println(higuchi);

      if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");

        String jsonData = "{";
        jsonData += "\"uuid\":\"" + uuid + "\",";
        jsonData += "\"datasetId\":\"" + datasetId + "\",";
        jsonData += "\"condition\":\"" + condition + "\",";
        jsonData += "\"hr\":" + String(heartRate) + ",";
        jsonData += "\"spo2\":" + String(SpO2) + ",";
        jsonData += "\"sys\":" + String(sysBP) + ",";
        jsonData += "\"dia\":" + String(diaBP) + ",";
        jsonData += "\"meanRR\":" + String(meanRR) + ",";
        jsonData += "\"medianRR\":" + String(medianRR) + ",";
        jsonData += "\"sdr\":" + String(sdr) + ",";
        jsonData += "\"rmssd\":" + String(rmssd) + ",";
        jsonData += "\"sdsd\":" + String(sdsd) + ",";
        jsonData += "\"sdr_rmssd\":" + String(sdr_rmssd) + ",";
        jsonData += "\"pnn25\":" + String(pnn25) + ",";
        jsonData += "\"pnn50\":" + String(pnn50) + ",";
        jsonData += "\"kurt\":" + String(kurt) + ",";
        jsonData += "\"skew\":" + String(skew) + ",";
        jsonData += "\"meanRelRR\":" + String(meanRelRR) + ",";
        jsonData += "\"medianRelRR\":" + String(medianRelRR) + ",";
        jsonData += "\"sdrRelRR\":" + String(sdrRelRR) + ",";
        jsonData += "\"rmssdRelRR\":" + String(rmssdRelRR) + ",";
        jsonData += "\"sdsdRelRR\":" + String(sdsdRelRR) + ",";
        jsonData += "\"sdrRmssdRelRR\":" + String(sdrRmssdRelRR) + ",";
        jsonData += "\"kurtRelRR\":" + String(kurtRelRR) + ",";
        jsonData += "\"skewRelRR\":" + String(skewRelRR) + ",";
        jsonData += "\"vlf\":" + String(vlf_power) + ",";
        jsonData += "\"vlf_pct\":" + String(vlf_pct) + ",";
        jsonData += "\"lf\":" + String(lf_power) + ",";
        jsonData += "\"lf_pct\":" + String(lf_pct) + ",";
        jsonData += "\"lf_nu\":" + String(lf_nu) + ",";
        jsonData += "\"hf\":" + String(hf_power) + ",";
        jsonData += "\"hf_pct\":" + String(hf_pct) + ",";
        jsonData += "\"hf_nu\":" + String(hf_nu) + ",";
        jsonData += "\"tp\":" + String(total_power) + ",";
        jsonData += "\"lf_hf\":" + String(lf_hf) + ",";
        jsonData += "\"hf_lf\":" + String(hf_lf) + ",";
        jsonData += "\"sd1\":" + String(sd1) + ",";
        jsonData += "\"sd2\":" + String(sd2) + ",";
        jsonData += "\"sampen\":" + String(sampen) + ",";
        jsonData += "\"higuchi\":" + String(higuchi) + ",";
        jsonData += "\"ecg\":" + String(ecgValue);
        jsonData += "}";

        int httpCode = http.POST(jsonData);
        if (httpCode > 0) {
          String response = http.getString();
          Serial.println(response);
        } else {
          Serial.println("Error in HTTP request");
        }
        http.end();
      }
    }
  }
}

// Implementations of helper functions (calculateMean, calculateMedian, etc.) remain unchanged from previous code.


// Implementations of helper functions (calculateMean, calculateMedian, etc.) remain unchanged from previous code.

// Calculate mean of RR intervals (long vector)
float calculateMean(const std::vector<long>& data) {
  long sum = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    sum += data[i];
  }
  return static_cast<float>(sum) / data.size();
}

// Calculate median of RR intervals (long vector)
float calculateMedian(const std::vector<long>& data) {
  std::vector<long> sortedData = data;
  std::sort(sortedData.begin(), sortedData.end());
  size_t size = sortedData.size();
  return size % 2 == 0 ? (sortedData[size / 2 - 1] + sortedData[size / 2]) / 2.0 : sortedData[size / 2];
}

// Calculate SDRR (Standard deviation of RR intervals) (long vector)
float calculateSDRR(const std::vector<long>& rrIntervals) {
  float mean = calculateMean(rrIntervals);
  float sumSquares = 0;
  for (size_t i = 0; i < rrIntervals.size(); ++i) {
    sumSquares += pow(rrIntervals[i] - mean, 2);
  }
  return sqrt(sumSquares / rrIntervals.size());
}

// Calculate RMSSD (Root mean square of successive RR interval differences) (long vector)
float calculateRMSSD(const std::vector<long>& rrIntervals) {
  float sumSquares = 0;
  for (size_t i = 1; i < rrIntervals.size(); ++i) {
    float diff = rrIntervals[i] - rrIntervals[i - 1];
    sumSquares += diff * diff;
  }
  return sqrt(sumSquares / (rrIntervals.size() - 1));
}

// Calculate SDSD (Standard deviation of successive RR interval differences) (long vector)
float calculateSDSD(const std::vector<long>& rrIntervals) {
  std::vector<float> diffRR;
  for (size_t i = 1; i < rrIntervals.size(); ++i) {
    diffRR.push_back(rrIntervals[i] - rrIntervals[i - 1]);
  }
  return calculateSDRRFloat(diffRR);
}

// Calculate pNN25 (Percentage of successive RR intervals that differ by more than 25 ms) (long vector)
float calculatePNN25(const std::vector<long>& rrIntervals) {
  int count = 0;
  for (size_t i = 1; i < rrIntervals.size(); ++i) {
    if (abs(rrIntervals[i] - rrIntervals[i - 1]) > 25) {
      count++;
    }
  }
  return (count * 100.0) / (rrIntervals.size() - 1);
}

// Calculate pNN50 (Percentage of successive RR intervals that differ by more than 50 ms) (long vector)
float calculatePNN50(const std::vector<long>& rrIntervals) {
  int count = 0;
  for (size_t i = 1; i < rrIntervals.size(); ++i) {
    if (abs(rrIntervals[i] - rrIntervals[i - 1]) > 50) {
      count++;
    }
  }
  return (count * 100.0) / (rrIntervals.size() - 1);
}

// Calculate Kurtosis (Kurtosis of the distribution of RR intervals) (long vector)
float calculateKurtosis(const std::vector<long>& rrIntervals) {
  float mean = calculateMean(rrIntervals);
  float sumFourthMoment = 0;
  float sumSquared = 0;
  
  for (size_t i = 0; i < rrIntervals.size(); ++i) {
    sumSquared += pow(rrIntervals[i] - mean, 2);
    sumFourthMoment += pow(rrIntervals[i] - mean, 4);
  }

  float variance = sumSquared / rrIntervals.size();
  float fourthMoment = sumFourthMoment / rrIntervals.size();
  
  return (fourthMoment / pow(variance, 2)) - 3;  // Excess Kurtosis
}

// Calculate Skewness (Skewness of the distribution of RR intervals) (long vector)
float calculateSkewness(const std::vector<long>& rrIntervals) {
  float mean = calculateMean(rrIntervals);
  float sumCubedMoment = 0;
  float sumSquared = 0;
  
  for (size_t i = 0; i < rrIntervals.size(); ++i) {
    sumSquared += pow(rrIntervals[i] - mean, 2);
    sumCubedMoment += pow(rrIntervals[i] - mean, 3);
  }

  float variance = sumSquared / rrIntervals.size();
  float skewness = (sumCubedMoment / rrIntervals.size()) / pow(variance, 1.5);
  return skewness;
}

// Calculate mean of float vector
float calculateMeanFloat(const std::vector<float>& data) {
  float sum = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    sum += data[i];
  }
  return sum / data.size();
}

// Calculate median of float vector
float calculateMedianFloat(const std::vector<float>& data) {
  std::vector<float> sortedData = data;
  std::sort(sortedData.begin(), sortedData.end());
  size_t size = sortedData.size();
  return size % 2 == 0 ? (sortedData[size / 2 - 1] + sortedData[size / 2]) / 2.0 : sortedData[size / 2];
}

// Calculate SDRR for float vector
float calculateSDRRFloat(const std::vector<float>& rrIntervals) {
  float mean = calculateMeanFloat(rrIntervals);
  float sumSquares = 0;
  for (size_t i = 0; i < rrIntervals.size(); ++i) {
    sumSquares += pow(rrIntervals[i] - mean, 2);
  }
  return sqrt(sumSquares / rrIntervals.size());
}

// Calculate RMSSD for float vector
float calculateRMSSDFloat(const std::vector<float>& rrIntervals) {
  float sumSquares = 0;
  for (size_t i = 1; i < rrIntervals.size(); ++i) {
    sumSquares += pow(rrIntervals[i] - rrIntervals[i - 1], 2);
  }
  return sqrt(sumSquares / (rrIntervals.size() - 1));
}

// Calculate SDSD for float vector
float calculateSDSDFloat(const std::vector<float>& rrIntervals) {
  std::vector<float> diffRR;
  for (size_t i = 1; i < rrIntervals.size(); ++i) {
    diffRR.push_back(rrIntervals[i] - rrIntervals[i - 1]);
  }
  return calculateSDRRFloat(diffRR);
}

// Calculate Kurtosis for float vector
float calculateKurtosisFloat(const std::vector<float>& rrIntervals) {
  float mean = calculateMeanFloat(rrIntervals);
  float sumFourthMoment = 0;
  float sumSquared = 0;
  
  for (size_t i = 0; i < rrIntervals.size(); ++i) {
    sumSquared += pow(rrIntervals[i] - mean, 2);
    sumFourthMoment += pow(rrIntervals[i] - mean, 4);
  }

  float variance = sumSquared / rrIntervals.size();
  float fourthMoment = sumFourthMoment / rrIntervals.size();
  return (fourthMoment / pow(variance, 2)) - 3;  // Excess Kurtosis
}

// Calculate Skewness for float vector
float calculateSkewnessFloat(const std::vector<float>& rrIntervals) {
  float mean = calculateMeanFloat(rrIntervals);
  float sumCubedMoment = 0;
  float sumSquared = 0;

  for (size_t i = 0; i < rrIntervals.size(); ++i) {
    sumSquared += pow(rrIntervals[i] - mean, 2);
    sumCubedMoment += pow(rrIntervals[i] - mean, 3);
  }

  float variance = sumSquared / rrIntervals.size();
  float skewness = (sumCubedMoment / rrIntervals.size()) / pow(variance, 1.5);
  return skewness;
}


// Calculate frequency-domain features (VLF, LF, HF)
float calculateFrequencyBandPower(const std::vector<long>& rrIntervals, float lowFreq, float highFreq) {
    float vReal[FFT_SIZE];
    float vImag[FFT_SIZE];

    // Fill vReal with RR intervals as float, pad with zeros if needed
    size_t n = rrIntervals.size();
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        if (i < n) {
            vReal[i] = (float)rrIntervals[i];
        } else {
            vReal[i] = 0.0f;
        }
        vImag[i] = 0.0f;
    }

    // Apply window
    fft.windowing(vReal, FFT_SIZE, FFTWindow::Hamming, FFTDirection::Forward);
    // Compute FFT
    fft.compute(vReal, vImag, FFT_SIZE, FFTDirection::Forward);
    // Convert to magnitude
    fft.complexToMagnitude(vReal, vImag, FFT_SIZE);

    // Calculate power in the desired frequency band
    float freqResolution = samplingFrequency / FFT_SIZE;
    float bandPower = 0.0f;
    for (size_t i = 0; i < FFT_SIZE / 2; ++i) {
        float freq = i * freqResolution;
        if (freq >= lowFreq && freq <= highFreq) {
            bandPower += vReal[i] * vReal[i];
        }
    }
    return bandPower;
}


float calculateVLF(const std::vector<long>& rrIntervals) {
  return calculateFrequencyBandPower(rrIntervals, 0.003, 0.04);
}

float calculateLF(const std::vector<long>& rrIntervals) {
  return calculateFrequencyBandPower(rrIntervals, 0.04, 0.15);
}

float calculateHF(const std::vector<long>& rrIntervals) {
  return calculateFrequencyBandPower(rrIntervals, 0.15, 0.4);
}

// SD1 & SD2 from Poincaré plot
float calculateSD1(const std::vector<long>& rrIntervals) {
  int N = rrIntervals.size();
  if (N < 2) return 0;
  std::vector<float> diff;
  for (int i = 0; i < N - 1; i++) {
    diff.push_back((rrIntervals[i+1] - rrIntervals[i]) / sqrt(2.0));
  }
  float mean = 0;
  for (float d : diff) mean += d;
  mean /= diff.size();
  float sum = 0;
  for (float d : diff) sum += pow(d - mean, 2);
  return sqrt(sum / diff.size());
}

float calculateSD2(const std::vector<long>& rrIntervals) {
  int N = rrIntervals.size();
  if (N < 2) return 0;

  float sd1 = calculateSD1(rrIntervals);
  float totalVar = 0;
  float meanRR = 0;
  for (int i = 0; i < N; i++) {
    meanRR += rrIntervals[i];
  }
  meanRR /= N;
  for (int i = 0; i < N; i++) {
    totalVar += pow(rrIntervals[i] - meanRR, 2);
  }
  totalVar /= N;
  return sqrt(2 * totalVar - pow(sd1, 2));
}

// Sample Entropy (simplified for small datasets)
float calculateSampen(const std::vector<long>& rrIntervals) {
  int m = 2;
  float r = 0.2 * stdDev(rrIntervals); // tolerance = 0.2 * SD
  int N = rrIntervals.size();
  if (N <= m + 1) return 0;

  int match_m = 0, match_m1 = 0;

  for (int i = 0; i < N - m - 1; i++) {
    for (int j = i + 1; j < N - m - 1; j++) {
      bool match = true;
      for (int k = 0; k < m; k++) {
        if (abs(rrIntervals[i + k] - rrIntervals[j + k]) > r) {
          match = false;
          break;
        }
      }
      if (match) {
        match_m++;
        if (abs(rrIntervals[i + m] - rrIntervals[j + m]) <= r) {
          match_m1++;
        }
      }
    }
  }

  if (match_m == 0 || match_m1 == 0) return -log(1.0 / N);
  return -log((float)match_m1 / match_m);
}

// Helper to compute standard deviation
float stdDev(const std::vector<long>& values) {
  float sum = 0, mean, SD = 0;
  int N = values.size();
  if (N == 0) return 0;
  for (auto v : values) sum += v;
  mean = sum / N;
  for (auto v : values) SD += pow(v - mean, 2);
  return sqrt(SD / N);
}

// Higuchi Fractal Dimension (HFD)
float calculateHiguchi(const std::vector<long>& rrIntervals) {
  int N = rrIntervals.size();
  int kmax = 5;
  if (N < kmax + 1) return 0;

  std::vector<float> L(kmax + 1);
  std::vector<float> lnL, lnK;

  for (int k = 1; k <= kmax; k++) {
    float Lk = 0;
    for (int m = 0; m < k; m++) {
      float len = 0;
      int count = 0;
      for (int i = m; i + k < N; i += k) {
        len += abs(rrIntervals[i + k] - rrIntervals[i]);
        count++;
      }
      float norm = (float)(N - 1) / (count * k);
      Lk += (len * norm) / k;
    }
    L[k] = Lk / k;
    lnL.push_back(log(L[k]));
    lnK.push_back(log(1.0 / k));
  }

  // Linear regression to get slope (HFD)
  float sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  int n = lnK.size();
  for (int i = 0; i < n; i++) {
    sumX += lnK[i];
    sumY += lnL[i];
    sumXY += lnK[i] * lnL[i];
    sumXX += lnK[i] * lnK[i];
  }

  float slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  return slope;
}
