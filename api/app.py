"""
Steam Revenue Prediction API
Flask server that runs in terminal (not notebook!)
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os

# ========================================================================
# CONFIGURATION
# ========================================================================

app = Flask(__name__)

# Model paths
MODEL_PATH = 'models/xgboost_revenue_model_tuned.pkl'
FEATURE_CONFIG_PATH = 'models/feature_names.json'

print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

# Load XGBoost model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✓ XGBoost model loaded")
else:
    print(f"❌ Model not found: {MODEL_PATH}")
    print("Please add xgboost_revenue_model_tuned.pkl to models/ folder")
    model = None

# Load feature names
if os.path.exists(FEATURE_CONFIG_PATH):
    with open(FEATURE_CONFIG_PATH, 'r') as f:
        FEATURE_ORDER = json.load(f)
    print(f"✓ Loaded {len(FEATURE_ORDER)} features")
else:
    print(f"❌ Feature config not found: {FEATURE_CONFIG_PATH}")
    FEATURE_ORDER = []

# Extract genres
GENRE_LIST = [f.replace('genre_', '') for f in FEATURE_ORDER if f.startswith('genre_')]
BASE_FEATURES = [f for f in FEATURE_ORDER if not f.startswith('genre_')]

print(f"✓ {len(GENRE_LIST)} genres, {len(BASE_FEATURES)} base features")
print("=" * 70)

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def prepare_features(data):
    """Convert input JSON to model features"""
    genre = data.get('genre', 'indie').lower().strip()
    
    if genre not in GENRE_LIST:
        raise ValueError(f'Invalid genre: "{genre}". Valid: {GENRE_LIST}')
    
    # Extract features with defaults
    features = {
        'final_price': float(data.get('final_price', 0)),
        'discount_pct_clean': float(data.get('discount_pct_clean', 0)),
        'is_on_sale': int(bool(data.get('is_on_sale', False))),
        'win_support': int(bool(data.get('win_support', True))),
        'mac_support': int(bool(data.get('mac_support', False))),
        'linux_support': int(bool(data.get('linux_support', False))),
        'multi_platform': int(bool(data.get('multi_platform', False))),
        'has_dlc': int(bool(data.get('has_dlc', False))),
        'dlc_available': int(data.get('dlc_available', 0)),
        'is_early_access': int(bool(data.get('is_early_access', False))),
        'is_free_to_play': int(bool(data.get('is_free_to_play', False)))
    }
    
    # One-hot encode genre
    for g in GENRE_LIST:
        features[f'genre_{g}'] = 1 if g == genre else 0
    
    # Create DataFrame with correct order
    df = pd.DataFrame([features])
    df = df[FEATURE_ORDER]
    
    return df, genre

# ========================================================================
# ROUTES
# ========================================================================

@app.route('/')
def home():
    """API documentation"""
    return jsonify({
        'api_name': 'Steam Revenue Prediction API',
        'version': '1.0',
        'status': 'Running in terminal (not notebook!)',
        'endpoints': {
            '/': 'This documentation',
            '/predict': 'POST - Predict revenue',
            '/health': 'GET - Health check',
            '/genres': 'GET - Available genres'
        },
        'example_request': {
            'url': 'POST http://localhost:5000/predict',
            'body': {
                'genre': 'rpg',
                'final_price': 480,
                'has_dlc': True,
                'dlc_available': 3
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict revenue"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please add model files to models/ folder'
        }), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        # Prepare features
        df, genre = prepare_features(data)
        
        # Predict
        prediction_log = model.predict(df)[0]
        revenue = np.expm1(prediction_log)
        
        return jsonify({
            'success': True,
            'predicted_revenue_rupees': float(revenue),
            'predicted_revenue_millions': round(float(revenue / 1e6), 2),
            'predicted_revenue_formatted': f"₹{revenue/1e6:.1f}M",
            'genre': genre,
            'input_data': data
        })
    
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': len(FEATURE_ORDER) > 0,
        'genres_available': len(GENRE_LIST)
    })

@app.route('/genres', methods=['GET'])
def genres():
    """List available genres"""
    return jsonify({
        'total_genres': len(GENRE_LIST),
        'genres': sorted(GENRE_LIST)
    })

# ========================================================================
# RUN SERVER
# ========================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 FLASK API SERVER STARTING")
    print("=" * 70)
    print("✓ Running in TERMINAL (not notebook!)")
    print("✓ Server URL: http://localhost:5000")
    print("✓ Press Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
