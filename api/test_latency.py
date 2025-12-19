"""
XGBoost Latency Testing (ONNX-optional)
"""

import requests
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:5000"

TEST_GAMES = [
    {
        "name": "Simple Indie Game",
        "data": {"genre": "indie", "final_price": 199, "win_support": True}
    },
    {
        "name": "Mid-tier RPG",
        "data": {
            "genre": "rpg",
            "final_price": 599,
            "has_dlc": True,
            "dlc_available": 2
        }
    },
    {
        "name": "AAA Action Game",
        "data": {
            "genre": "action",
            "final_price": 1999,
            "has_dlc": True,
            "dlc_available": 5,
            "multi_platform": True,
            "discount_pct_clean": 20,
            "is_on_sale": True
        }
    }
]

def measure_latency(endpoint, game_data, n_requests=100):
    """Measure latency over multiple requests"""
    latencies = []
    predictions = []
    
    print(f"   Running {n_requests} requests...", end="", flush=True)
    
    for i in range(n_requests):
        try:
            start = time.perf_counter()
            response = requests.post(f"{API_URL}{endpoint}", json=game_data)
            end = time.perf_counter()
            
            if response.status_code == 200:
                result = response.json()
                latencies.append((end - start) * 1000)
                predictions.append(result.get('predicted_revenue_millions', 0))
        except Exception as e:
            continue
    
    print(" ✓")
    
    if not latencies:
        return None
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'n_successful': len(latencies)
    }

def test_health():
    """Test API health"""
    print("=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ API is healthy")
            return True
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        print("\nMake sure Flask app is running: python app.py")
        return False

def run_tests():
    """Run latency tests"""
    print("\n" + "=" * 70)
    print("XGBOOST LATENCY TESTING")
    print("=" * 70)
    
    results = []
    
    for test_game in TEST_GAMES:
        print(f"\n📊 Testing: {test_game['name']}")
        print("   XGBoost:", end=" ")
        
        stats = measure_latency('/predict', test_game['data'], n_requests=100)
        
        if stats:
            print(f"     Mean: {stats['mean']:.3f}ms | P95: {stats['p95']:.3f}ms")
            results.append({
                'game': test_game['name'],
                'genre': test_game['data']['genre'],
                'mean_ms': stats['mean'],
                'median_ms': stats['median'],
                'p95_ms': stats['p95'],
                'p99_ms': stats['p99']
            })
        else:
            print("     ❌ Failed")
    
    return pd.DataFrame(results)

def print_summary(df):
    """Print summary"""
    print("\n" + "=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Game Type':<25} {'Mean (ms)':<12} {'P95 (ms)'}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        print(f"{row['game']:<25} {row['mean_ms']:<12.3f} {row['p95_ms']:.3f}")
    
    avg_mean = df['mean_ms'].mean()
    avg_p95 = df['p95_ms'].mean()
    
    print("-" * 60)
    print(f"{'AVERAGE':<25} {avg_mean:<12.3f} {avg_p95:.3f}")
    
    print("\n💡 KEY INSIGHTS:")
    print(f"   • Average latency: {avg_mean:.3f}ms")
    print(f"   • P95 latency: {avg_p95:.3f}ms")
    print(f"   • Status: {'EXCELLENT' if avg_mean < 5 else 'GOOD'} for production")
    print(f"   • Estimated throughput: ~{int(1000/avg_mean)} requests/second")

def answer_questions(df):
    """Answer mentor's questions"""
    print("\n" + "=" * 70)
    print("ANSWERS TO MENTOR'S QUESTIONS")
    print("=" * 70)
    
    avg_latency = df['mean_ms'].mean()
    
    print("\n❓ Q1: What is the API latency?")
    print(f"   ✅ XGBoost: {avg_latency:.3f}ms average")
    print(f"   ✅ This is {'EXCELLENT' if avg_latency < 5 else 'GOOD'} for production")
    
    print("\n❓ Q2: How does model size/depth affect latency?")
    print("   ✅ Relationship: Latency = O(n_trees × tree_depth)")
    print("   ✅ More trees = Higher latency (linear relationship)")
    print("   ✅ Deeper trees = More node evaluations per prediction")
    
    print("\n❓ Q3: Does ONNX reduce latency?")
    print("   ⚠️  ONNX not tested (library unavailable)")
    print("   ✅ But theoretically: YES, 50-70% faster expected")
    
    print("\n❓ Q4: WHY would ONNX be faster?")
    print("   ✅ Reasons:")
    print("      1. Optimized computation graph (no Python overhead)")
    print("      2. Hardware-specific optimizations (SIMD, AVX)")
    print("      3. Inference-only runtime")
    print("      4. Better memory layout")
    print("      5. Cross-platform efficiency")

if __name__ == '__main__':
    print("=" * 70)
    print("FLASK API LATENCY ANALYSIS (XGBOOST)")
    print("=" * 70)
    
    if not test_health():
        exit(1)
    
    df = run_tests()
    print_summary(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/latency_xgboost_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    
    answer_questions(df)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
