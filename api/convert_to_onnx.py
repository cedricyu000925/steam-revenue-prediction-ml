"""
XGBoost Model Analysis (ONNX-optional)
Analyzes model performance without ONNX conversion
"""

import joblib
import json
import os
import time
import numpy as np
import pandas as pd

print("=" * 70)
print("XGBOOST MODEL ANALYSIS")
print("=" * 70)

# Load XGBoost model
print("\n1️⃣  Loading XGBoost model...")
model = joblib.load('models/xgboost_revenue_model_tuned.pkl')
print(f"   ✓ Model loaded: {type(model).__name__}")

# Get model parameters
params = model.get_params()
print(f"\n📊 Model Configuration:")
print(f"   n_estimators (trees): {params.get('n_estimators', 'N/A')}")
print(f"   max_depth: {params.get('max_depth', 'N/A')}")
print(f"   learning_rate: {params.get('learning_rate', 'N/A')}")

# Load features
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)
n_features = len(feature_names)
print(f"   Features: {n_features}")

# Check model size
pkl_size = os.path.getsize('models/xgboost_revenue_model_tuned.pkl') / 1024
print(f"\n2️⃣  Model Size:")
print(f"   XGBoost (.pkl): {pkl_size:.2f} KB")

# Test prediction speed with various input sizes
print(f"\n3️⃣  Testing Prediction Latency...")

# Create test data
dummy_input = pd.DataFrame(
    np.random.rand(100, n_features),
    columns=feature_names
)

# Warm up (first predictions are slower)
print("   Warming up model...")
for _ in range(20):
    _ = model.predict(dummy_input[:1])

# Measure single-prediction latency
print("   Measuring single-prediction latency...")
latencies = []
for i in range(100):
    start = time.perf_counter()
    _ = model.predict(dummy_input[i:i+1])
    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # Convert to ms

avg_latency = np.mean(latencies)
median_latency = np.median(latencies)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)
min_latency = np.min(latencies)
max_latency = np.max(latencies)

print(f"\n   Results (100 predictions):")
print(f"   • Mean:   {avg_latency:.3f}ms")
print(f"   • Median: {median_latency:.3f}ms")
print(f"   • P95:    {p95_latency:.3f}ms")
print(f"   • P99:    {p99_latency:.3f}ms")
print(f"   • Min:    {min_latency:.3f}ms")
print(f"   • Max:    {max_latency:.3f}ms")

# Calculate throughput
throughput = 1000 / avg_latency
print(f"\n   Throughput: ~{throughput:.0f} predictions/second")

# Analyze impact of model complexity
print(f"\n4️⃣  Model Complexity Analysis:")
n_trees = params.get('n_estimators', 300)
depth = params.get('max_depth', 6)

print(f"\n   Current Configuration:")
print(f"   • Trees: {n_trees}")
print(f"   • Depth: {depth}")
print(f"   • Latency: {avg_latency:.3f}ms")

# Estimate latency for different configurations
print(f"\n   Estimated Latency by Configuration:")
print(f"   {'Trees':<8} {'Depth':<8} {'Est. Latency'}")
print(f"   {'-'*35}")

# Base ratio
base_complexity = n_trees * depth
base_latency = avg_latency

configs = [
    (100, 3),
    (100, 6),
    (200, 5),
    (300, 6),  # Current
    (500, 6),
    (500, 10),
]

for trees, d in configs:
    complexity = trees * d
    est_latency = base_latency * (complexity / base_complexity)
    marker = " ← CURRENT" if trees == n_trees and d == depth else ""
    print(f"   {trees:<8} {d:<8} {est_latency:.3f}ms{marker}")

print(f"\n   💡 Insight: Latency scales with (n_trees × depth)")

# Save results
print(f"\n5️⃣  Saving Results...")
results = {
    'model_type': str(type(model).__name__),
    'n_estimators': n_trees,
    'max_depth': depth,
    'n_features': n_features,
    'model_size_kb': pkl_size,
    'mean_latency_ms': avg_latency,
    'median_latency_ms': median_latency,
    'p95_latency_ms': p95_latency,
    'p99_latency_ms': p99_latency,
    'throughput_per_sec': throughput
}

results_df = pd.DataFrame([results])
results_df.to_csv('results/model_analysis.csv', index=False)
print(f"   ✓ Results saved to: results/model_analysis.csv")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)

print("\n📋 Summary:")
print(f"   • XGBoost latency: {avg_latency:.3f}ms (excellent for production)")
print(f"   • Model size: {pkl_size:.2f} KB")
print(f"   • Throughput: ~{throughput:.0f} req/sec")
print(f"   • Complexity: {n_trees} trees × depth {depth}")
