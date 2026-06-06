import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.predictor import EduPulsePredictor

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# In-memory model cache: {dataset: {model_name: predictor}}
PREDICTORS = {}
MODEL_NAMES = ['catboost', 'xgboost', 'random_forest', 'mlp', 'svm']
DATASETS = ['uci', 'oulad']
COMPARISON_PATH = os.path.join(BASE_DIR, 'models', 'model_comparison.json')


def preload_models():
    """Load all available trained models at startup."""
    print("Preloading models...")
    for ds in DATASETS:
        PREDICTORS[ds] = {}
        for mn in MODEL_NAMES:
            try:
                p = EduPulsePredictor()
                p.load_model(ds, mn)
                PREDICTORS[ds][mn] = p
                print(f"  ✓ {ds}/{mn}")
            except FileNotFoundError:
                print(f"  ✗ {ds}/{mn} (not trained yet)")
            except Exception as e:
                print(f"  ✗ {ds}/{mn}: {e}")


@app.route('/health')
def health():
    return jsonify({"status": "ok", "models_loaded": sum(len(v) for v in PREDICTORS.values())})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models')
def get_models():
    """Return all available models with their accuracies."""
    comparison = {}
    if os.path.exists(COMPARISON_PATH):
        try:
            with open(COMPARISON_PATH) as f:
                comparison = json.load(f)
        except Exception:
            pass

    result = []
    for mn in MODEL_NAMES:
        entry = {
            "id": mn,
            "name": mn.replace('_', ' ').title(),
            "uci_accuracy": 0.0,
            "oulad_accuracy": 0.0,
            "uci_f1": 0.0,
            "oulad_f1": 0.0
        }
        for ds in DATASETS:
            if ds in comparison and mn in comparison[ds]:
                entry[f"{ds}_accuracy"] = round(comparison[ds][mn].get('accuracy', 0) * 100, 2)
                entry[f"{ds}_f1"] = round(comparison[ds][mn].get('f1_score', 0), 3)
        result.append(entry)
    return jsonify(result)


@app.route('/model-comparison')
def model_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return jsonify({"error": "model_comparison.json not found. Train models first."}), 404
    try:
        with open(COMPARISON_PATH) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Single student prediction endpoint."""
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "JSON body required"}), 400

        dataset = body.get('dataset', '').lower()
        model_name = body.get('model', '').lower()
        data = body.get('data', {})

        if dataset not in DATASETS:
            return jsonify({"error": f"'dataset' must be one of {DATASETS}"}), 400
        if model_name not in MODEL_NAMES:
            return jsonify({"error": f"'model' must be one of {MODEL_NAMES}"}), 400
        if not isinstance(data, dict):
            return jsonify({"error": "'data' must be a JSON object of feature values"}), 400

        predictor = PREDICTORS.get(dataset, {}).get(model_name)
        if predictor is None:
            return jsonify({"error": f"Model {dataset}/{model_name} not loaded. Run trainer.py first."}), 503

        result = predictor.predict(data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV upload."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded. Use form field 'file'"}), 400

        file = request.files['file']
        dataset = request.form.get('dataset', 'uci').lower()
        model_name = request.form.get('model', 'catboost').lower()

        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files supported"}), 400
        if dataset not in DATASETS:
            return jsonify({"error": f"'dataset' must be one of {DATASETS}"}), 400
        if model_name not in MODEL_NAMES:
            return jsonify({"error": f"'model' must be one of {MODEL_NAMES}"}), 400

        try:
            df = pd.read_csv(file)
            if len(df.columns) <= 1:
                file.seek(0)
                df = pd.read_csv(file, sep=';')
        except Exception as e:
            return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

        if len(df) == 0:
            return jsonify({"error": "Uploaded CSV is empty"}), 400

        predictor = PREDICTORS.get(dataset, {}).get(model_name)
        if predictor is None:
            return jsonify({"error": f"Model {dataset}/{model_name} not loaded"}), 503

        predictions = []
        for idx, row in df.iterrows():
            row_dict = {k: (v.item() if hasattr(v, 'item') else v)
                        for k, v in row.to_dict().items()
                        if not (isinstance(v, float) and np.isnan(v))}
            try:
                result = predictor.predict(row_dict)
                predictions.append({"row": idx + 1, **result})
            except Exception as e:
                predictions.append({"row": idx + 1, "error": str(e)})

        return jsonify({
            "dataset": dataset,
            "model": model_name,
            "total_rows": len(df),
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": f"Batch prediction error: {str(e)}"}), 500


if __name__ == '__main__':
    preload_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
