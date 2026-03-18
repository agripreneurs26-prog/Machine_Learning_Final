from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Get absolute path (FIXES Render FileNotFoundError)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load model & scaler safely
try:
    model_path = os.path.join(BASE_DIR, 'optimized_xgboost.joblib')
    scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("✅ Model and scaler loaded successfully")

except Exception as e:
    print("❌ Error loading model/scaler:", e)
    model = None
    scaler = None


# ✅ Expected input columns
expected_columns = [
    'gender', 'region', 'highest_education', 'imd_band', 'age_band',
    'num_of_prev_attempts', 'studied_credits', 'score'
]

# ✅ Columns to scale
features_to_scale = ['num_of_prev_attempts', 'studied_credits', 'score']


# ✅ Health check route (IMPORTANT for Render)
@app.route('/')
def home():
    return "🚀 API is running successfully!"


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check model loaded
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Model or scaler not loaded'
            })

        data = request.get_json()

        # ✅ Handle empty request
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            })

        # ✅ Validate input fields
        for col in expected_columns:
            if col not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing field: {col}'
                })

        # ✅ Convert to DataFrame
        input_df = pd.DataFrame([data])
        input_df = input_df[expected_columns]

        # ✅ Convert numeric safely
        input_df[features_to_scale] = input_df[features_to_scale].astype(float)

        # ✅ Scale features
        input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])

        # ✅ Prediction
        prediction = model.predict(input_df)

        return jsonify({
            'status': 'success',
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


