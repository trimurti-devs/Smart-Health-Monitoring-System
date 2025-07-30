from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
VERSION = "2.1.0"

# === ML Model & Scaler Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FALL_MODEL_PATH = os.getenv("FALL_MODEL_PATH", os.path.join(BASE_DIR, "fall_detection_model.pkl"))
FALL_SCALER_PATH = os.getenv("FALL_SCALER_PATH", os.path.join(BASE_DIR, "fall_detection_scaler.pkl"))
STRESS_MODEL_PATH = os.getenv("STRESS_MODEL_PATH", os.path.join(BASE_DIR, "stress_model.pkl"))
STRESS_SCALER_PATH = os.getenv("STRESS_SCALER_PATH", os.path.join(BASE_DIR, "stress_scaler.pkl"))
# Remove immobility model and scaler paths since no immobility pkl files
# IMMOBILITY_MODEL_PATH = os.getenv("IMMOBILITY_MODEL_PATH", os.path.join(BASE_DIR, "immobility_detection_model.pkl"))
# IMMOBILITY_SCALER_PATH = os.getenv("IMMOBILITY_SCALER_PATH", os.path.join(BASE_DIR, "immobility_detection_scaler.pkl"))

logging.info(f"🔍 Loading API v{VERSION}")
logging.info(f"📁 Fall Model path: {FALL_MODEL_PATH}")
logging.info(f"📁 Stress Model path: {STRESS_MODEL_PATH}")
# logging.info(f"📁 Immobility Model path: {IMMOBILITY_MODEL_PATH}")

# === Load Fall & Stress Models & Scalers ===
def load_model_and_scaler(model_path, scaler_path):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logging.info(f"✅ Loaded model from {model_path}")
            return model, scaler
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
    return None, None

fall_model, fall_scaler = load_model_and_scaler(FALL_MODEL_PATH, FALL_SCALER_PATH)
stress_model, stress_scaler = load_model_and_scaler(STRESS_MODEL_PATH, STRESS_SCALER_PATH)
# Remove immobility model and scaler loading
# immobility_model, immobility_scaler = load_model_and_scaler(IMMOBILITY_MODEL_PATH, IMMOBILITY_SCALER_PATH)

if stress_scaler is not None and hasattr(stress_scaler, 'feature_names_in_'):
    logging.info(f"ℹ Stress scaler feature names: {stress_scaler.feature_names_in_}")

# === Model Training Function (Fallback) ===
def train_and_save_model():
    logging.warning("🔄 Retraining models...")
    
    # Fall detection training (Dummy example)
    X_train_fall = np.random.rand(100, 9)
    y_train_fall = np.random.randint(0, 2, size=100)
    scaler_fall = StandardScaler().fit(X_train_fall)
    model_fall = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler_fall.transform(X_train_fall), y_train_fall)
    joblib.dump(model_fall, FALL_MODEL_PATH)
    joblib.dump(scaler_fall, FALL_SCALER_PATH)
    logging.info("✅ Fall detection model & scaler saved!")

    # Stress prediction training (Dummy example)
    X_train_stress = np.random.rand(100, 9)
    y_train_stress = np.random.randint(0, 2, size=100)
    scaler_stress = StandardScaler().fit(X_train_stress)
    model_stress = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler_stress.transform(X_train_stress), y_train_stress)
    joblib.dump(model_stress, STRESS_MODEL_PATH)
    joblib.dump(scaler_stress, STRESS_SCALER_PATH)
    logging.info("✅ Stress prediction model & scaler saved!")

# === Load Fall & Stress Models & Scalers ===
def load_model_and_scaler(model_path, scaler_path):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logging.info(f"✅ Loaded model from {model_path}")
            return model, scaler
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
    return None, None

fall_model, fall_scaler = load_model_and_scaler(FALL_MODEL_PATH, FALL_SCALER_PATH)
stress_model, stress_scaler = load_model_and_scaler(STRESS_MODEL_PATH, STRESS_SCALER_PATH)

if fall_model is None or fall_scaler is None or stress_model is None or stress_scaler is None:
    logging.warning("⚠ Missing models/scalers. Retraining...")
    train_and_save_model()
    fall_model, fall_scaler = load_model_and_scaler(FALL_MODEL_PATH, FALL_SCALER_PATH)
    stress_model, stress_scaler = load_model_and_scaler(STRESS_MODEL_PATH, STRESS_SCALER_PATH)

# === Routes ===

@app.route('/')
def index():
    return '🧠 ESP32 Health + Fall & Stress Detection Server running!'

@app.route('/version', methods=['GET'])
def get_version():
    return jsonify({"version": VERSION, "status": "API is running"})

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.json
    logging.info("📥 Data received from ESP32: %s", data)
    logging.info(f"🔑 Incoming data keys: {list(data.keys())}")

    # Log fall detection status if present
    if 'fall_detected' in data:
        logging.info(f"⚠️ Fall Detected Status: {data['fall_detected']}")

    # Emit to web dashboard via SocketIO
    socketio.emit('sensor_data', data)

    # Map incoming keys to expected uppercase keys for stress prediction
    key_map = {
        "meanRR": "MEAN_RR",
        "medianRR": "MEDIAN_RR",
        "hr": "HR",
        "meanRelRR": "MEAN_REL_RR",
        "medianRelRR": "MEDIAN_REL_RR",
        "sdr": "SDRR",
        "sdrRelRR": "SDRR_REL_RR",
        "rmssd": "RMSSD",
        "rmssdRelRR": "RMSSD_REL_RR",
        "sdsd": "SDSD",
        "sdsdRelRR": "SDSD_REL_RR",
        "sdr_rmssd": "SDRR_RMSSD",
        "sdrRmssdRelRR": "SDRR_RMSSD_REL_RR",
        "pnn25": "pNN25",
        "pnn50": "pNN50",
        "kurt": "KURT",
        "kurtRelRR": "KURT_REL_RR",
        "skew": "SKEW",
        "skewRelRR": "SKEW_REL_RR",
        "sd1": "SD1",
        "sd2": "SD2",
        "sampen": "sampen",
        "higuchi": "higuci",
        "vlf": "VLF",
        "vlf_pct": "VLF_PCT",
        "lf": "LF",
        "lf_pct": "LF_PCT",
        "lf_nu": "LF_NU",
        "hf": "HF",
        "hf_pct": "HF_PCT",
        "hf_nu": "HF_NU",
        "tp": "TP",
        "lf_hf": "LF_HF",
        "hf_lf": "HF_LF"
    }

    # Create mapped data dict for prediction
    mapped_data = {}
    for k_in, k_out in key_map.items():
        if k_in in data:
            mapped_data[k_out] = data[k_in]

    logging.info("🔄 Mapped data for stress prediction: %s", mapped_data)

    # Automatically predict stress from received data if possible
    if stress_model is not None and stress_scaler is not None:
        try:
            required_keys = stress_scaler.feature_names_in_ if hasattr(stress_scaler, 'feature_names_in_') else []
            if required_keys is not None and len(required_keys) > 0:
                missing = [k for k in required_keys if k not in mapped_data]
                if len(missing) == 0:
                    features = np.array([[float(mapped_data[k]) for k in required_keys]])
                    features_scaled = stress_scaler.transform(features)
                    prediction = stress_model.predict(features_scaled)[0]
                    # Map prediction to multi-class labels
                    label_map = {0: "no stress", 1: "interruption", 2: "time pressure"}
                    result = label_map.get(prediction, "Unknown")
                    proba = stress_model.predict_proba(features_scaled)[0]
                    confidence = round(float(proba[prediction]), 4)

                    stress_result = {
                        "prediction": result,
                        "raw_output": int(prediction),
                        "confidence": confidence,
                        "probabilities": proba.tolist()
                    }
                    logging.info(f"📤 Auto Stress Prediction Result: {stress_result}")

                    # Emit stress prediction via SocketIO for real-time update
                    socketio.emit('stress_prediction', stress_result)
                else:
                    logging.warning(f"⚠ Missing keys for stress prediction: {missing}")
            else:
                logging.warning("⚠ Stress scaler does not have feature names info.")
        except Exception as e:
            logging.error(f"❌ Auto Stress Prediction Error: {e}")
    else:
        logging.warning("⚠ Stress model or scaler not loaded, skipping stress prediction.")

    return jsonify({"status": "success"}), 200

@app.route('/predict_fall', methods=['POST'])
def predict_fall():
    if fall_model is None or fall_scaler is None:
        return jsonify({"error": "Fall model or scaler not loaded"}), 500

    try:
        data = request.get_json()
        logging.info(f"📥 Fall Prediction Data: {data}")

        required_keys = [
            "acc_max", "gyro_max", "acc_kurtosis", "gyro_kurtosis",
            "lin_max", "acc_skewness", "gyro_skewness",
            "post_gyro_max", "post_lin_max"
        ]
        missing = [k for k in required_keys if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = np.array([[float(data[k]) for k in required_keys]])
        features_scaled = fall_scaler.transform(features)
        prediction = fall_model.predict(features_scaled)[0]
        result = "Fall Detected" if prediction == 1 else "No Fall"
        confidence = round(float(fall_model.predict_proba(features_scaled)[0][prediction]), 4)

        response = {
            "version": VERSION,
            "prediction": result,
            "raw_output": int(prediction),
            "confidence": confidence
        }
        logging.info(f"📤 Fall Prediction Result: {response}")

        # Emit fall prediction via SocketIO for real-time update
        socketio.emit('fall_prediction', response)

        return jsonify(response)

    except Exception as e:
        logging.error(f"❌ Fall Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_stress', methods=['POST'])
def predict_stress():
    if stress_model is None or stress_scaler is None:
        return jsonify({"error": "Stress model or scaler not loaded"}), 500

    try:
        data = request.get_json()
        logging.info(f"📥 Stress Prediction Data: {data}")

        required_keys = [
            "MEAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD",
            "pNN25", "pNN50", "KURT", "SKEW"
        ]
        missing = [k for k in required_keys if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = np.array([[float(data[k]) for k in required_keys]])
        features_scaled = stress_scaler.transform(features)
        prediction = stress_model.predict(features_scaled)[0]
        result = "Stressed" if prediction == 1 else "Not Stressed"
        confidence = round(float(stress_model.predict_proba(features_scaled)[0][prediction]), 4)

        response = {
            "version": VERSION,
            "prediction": result,
            "raw_output": int(prediction),
            "confidence": confidence
        }
        logging.info(f"📤 Stress Prediction Result: {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"❌ Stress Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# Remove immobility prediction route and replace with immobility prediction using fall model
@app.route('/predict_immobility', methods=['POST'])
def predict_immobility():
    if fall_model is None or fall_scaler is None:
        return jsonify({"error": "Fall model or scaler not loaded"}), 500

    try:
        data = request.get_json()
        logging.info(f"📥 Immobility Prediction Data (using fall model): {data}")

        # Use the same required keys as fall detection
        required_keys = [
            "acc_max", "gyro_max", "acc_kurtosis", "gyro_kurtosis",
            "lin_max", "acc_skewness", "gyro_skewness",
            "post_gyro_max", "post_lin_max"
        ]
        missing = [k for k in required_keys if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = np.array([[float(data[k]) for k in required_keys]])
        features_scaled = fall_scaler.transform(features)
        prediction = fall_model.predict(features_scaled)[0]

        # Define immobility detection logic based on fall model prediction or other heuristic
        # For example, if no fall detected and some other condition, immobility detected
        # Here, as a placeholder, immobility detected if no fall (prediction == 0)
        immobility_detected = prediction == 0
        result = "Immobility Detected" if immobility_detected else "No Immobility"
        confidence = round(float(fall_model.predict_proba(features_scaled)[0][prediction]), 4)

        response = {
            "version": VERSION,
            "prediction": result,
            "raw_output": int(immobility_detected),
            "confidence": confidence
        }
        logging.info(f"📤 Immobility Prediction Result (using fall model): {response}")

        # Emit immobility prediction via SocketIO for real-time update
        socketio.emit('immobility_prediction', response)

        return jsonify(response)

    except Exception as e:
        logging.error(f"❌ Immobility Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# === Run Server ===
if __name__ == '__main__':
    logging.info("🚀 Flask server running at http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
