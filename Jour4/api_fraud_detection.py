"""
API Flask - Détection de fraude bancaire
TP4 - Master 1 Data Engineering
Endpoints: POST /predict, GET /health
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Charger le modèle au démarrage
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fraud_detector_v1.joblib')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'metadata_v1.joblib')

model = None
metadata = None
threshold = 0.5


def load_model():
    global model, metadata, threshold
    try:
        model = joblib.load(MODEL_PATH)
        if os.path.exists(METADATA_PATH):
            metadata = joblib.load(METADATA_PATH)
            threshold = metadata.get('optimal_threshold', 0.5)
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")


def get_risk_level(probability):
    if probability < 0.3:
        return 'low'
    elif probability < 0.6:
        return 'medium'
    elif probability < 0.85:
        return 'high'
    else:
        return 'critical'


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_version': metadata.get('model_version', '1.0') if metadata else '1.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 503
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON vide'}), 400
        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[:, 1][0]
        is_fraud = proba >= threshold
        return jsonify({
            'is_fraud': bool(is_fraud),
            'probability': float(proba),
            'risk_level': get_risk_level(proba)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
