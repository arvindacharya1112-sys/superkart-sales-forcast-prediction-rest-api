
import os
import io
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# Initialize Flask App and Enable CORS
superkart_sales_predictor_api = Flask("SuperKart Sales Forecast Predictor")
CORS(superkart_sales_predictor_api)
app = superkart_sales_predictor_api

# ==========================================
# MODEL LOADING LOGIC
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'superkart_sales_forecast_prediction_model_v1_0.joblib'

# Search both backend_files/ and the root folder for the model file
model_path_in_folder = os.path.join(BASE_DIR, MODEL_NAME)
model_path_in_root = os.path.join(os.path.dirname(BASE_DIR), MODEL_NAME)

if os.path.exists(model_path_in_folder):
    MODEL_PATH = model_path_in_folder
elif os.path.exists(model_path_in_root):
    MODEL_PATH = model_path_in_root
else:
    MODEL_PATH = MODEL_NAME

print(f"🔍 Searching for model at: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR: Failed to load model file at {MODEL_PATH}")
    print(f"Error details: {e}")
    model = None

# ==========================================
# FEATURE DEFINITIONS & PREPROCESSING
# ==========================================
MODEL_FEATURES = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Id_char',
    'Product_Type_Category',
    'Store_Age_Years',
    'Product_Allocated_Area_Log',
]


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Computes missing transformed features if missing from payload."""
    df = df.copy()

    # 1. Compute Log transform for Product_Allocated_Area if needed
    if (
        'Product_Allocated_Area_Log' not in df.columns
        and 'Product_Allocated_Area' in df.columns
    ):
        df['Product_Allocated_Area_Log'] = np.log1p(df['Product_Allocated_Area'])

    # 2. Assign default or mapped Category if missing
    if 'Product_Type_Category' not in df.columns:
        if 'Product_Type' in df.columns:
            df['Product_Type_Category'] = df['Product_Type']
        else:
            df['Product_Type_Category'] = 'General'

    # Ensure return dataframe contains only required features in correct order
    return df[MODEL_FEATURES]


# ==========================================
# 1. SINGLE PREDICTION ENDPOINT
# ==========================================
@superkart_sales_predictor_api.post('/v1/predict')
def predict_superkart_sales():
    try:
        if model is None:
            return jsonify({'error': 'Model is not loaded on server.'}), 500

        superkart_data = request.get_json()
        if not superkart_data:
            return jsonify({'error': 'No JSON payload provided'}), 400

        input_df = pd.DataFrame([superkart_data])
        processed_df = preprocess_input(input_df)

        raw_prediction = model.predict(processed_df)[0]
        predicted_sales = round(float(raw_prediction), 2)

        return jsonify({'Predicted Sales (in dollars)': predicted_sales}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==========================================
# 2. BATCH PREDICTION ENDPOINT
# ==========================================
@superkart_sales_predictor_api.post('/v1/predict-batch')
def predict_batch_superkart_sales():
    try:
        if model is None:
            return jsonify({'error': 'Model is not loaded on server.'}), 500

        # Check for uploaded file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename uploaded'}), 400

        # Read CSV file into pandas DataFrame
        input_df = pd.read_csv(file)
        processed_df = preprocess_input(input_df)

        # Generate predictions for all rows
        raw_predictions = model.predict(processed_df)

        # Build ID/Index -> Prediction map expected by Streamlit
        results = {}
        for idx, pred in enumerate(raw_predictions):
            row_identifier = str(input_df.iloc[idx].get('Product_Id', idx))
            results[row_identifier] = round(float(pred), 2)

        return jsonify(results), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==========================================
# SERVER ENTRY POINT
# ==========================================
if __name__ == '__main__':
    print("🚀 Starting Flask server on http://0.0.0.0:7860 ...")
    app.run(host='0.0.0.0', port=7860, debug=True)
