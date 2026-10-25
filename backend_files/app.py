import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask application
app = Flask(__name__)

# Dynamically resolve path to the serialized pipeline relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'superkart_model.joblib')

# Load the model pipeline
model = joblib.load(MODEL_PATH)

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        # Extract features from JSON payload
        data = request.get_json(force=True)

        # Convert inputs to DataFrame as expected by the pipeline
        input_df = pd.DataFrame([data])

        # Predict using pipeline
        prediction = model.predict(input_df)[0]

        return jsonify({
            'status': 'success',
            'prediction': float(prediction)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        # Check if file is provided in request
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected for uploading'}), 400

        # Load the CSV data into DataFrame
        input_df = pd.read_csv(file)

        # Predict for all records
        predictions = model.predict(input_df)

        # Return indices with corresponding predictions in JSON format
        results = {str(idx): float(pred) for idx, pred in enumerate(predictions)}
        return jsonify(results), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
