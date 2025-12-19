"""
Check actual XGBoost model parameters
"""

import joblib
import json
import os

print("=" * 70)
print("CHECKING YOUR MODEL CONFIGURATION")
print("=" * 70)

# Load your model
print("\n1️⃣  Loading model...")
model = joblib.load('models/xgboost_revenue_model_tuned.pkl')
print(f"   ✓ Model type: {type(model).__name__}")

# Get ACTUAL parameters
print("\n2️⃣  Model Parameters:")
params = model.get_params()

# Display all parameters
for key, value in params.items():
    if value is not None and value != "deprecated":
        print(f"   • {key}: {value}")

# Key parameters
print("\n3️⃣  Key Configuration:")
print(f"   n_estimators (trees): {params.get('n_estimators', 'N/A')}")
print(f"   max_depth: {params.get('max_depth', 'N/A')}")
print(f"   learning_rate: {params.get('learning_rate', 'N/A')}")
print(f"   subsample: {params.get('subsample', 'N/A')}")
print(f"   colsample_bytree: {params.get('colsample_bytree', 'N/A')}")

# Load features
print("\n4️⃣  Features:")
with open('models/feature_names.json', 'r') as f:
    features = json.load(f)
print(f"   Total features: {len(features)}")
print(f"   Features: {features}")

# Model size
print("\n5️⃣  Model File Size:")
pkl_size = os.path.getsize('models/xgboost_revenue_model_tuned.pkl')
print(f"   Size (bytes): {pkl_size:,}")
print(f"   Size (KB): {pkl_size / 1024:.2f} KB")
print(f"   Size (MB): {pkl_size / (1024*1024):.2f} MB")

# Check if model has additional info
print("\n6️⃣  Additional Model Info:")
if hasattr(model, 'n_features_in_'):
    print(f"   Features expected: {model.n_features_in_}")
if hasattr(model, 'feature_names_in_'):
    print(f"   Feature names stored: {len(model.feature_names_in_)}")

print("\n" + "=" * 70)
print("✅ MODEL CHECK COMPLETE")
print("=" * 70)
