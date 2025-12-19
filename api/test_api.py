"""
Test the Flask API
Run this AFTER starting app.py
"""

import requests
import json
import time

API_URL = "http://localhost:5000"

print("=" * 70)
print("TESTING STEAM REVENUE API")
print("=" * 70)

# Test 1: Health Check
print("\n1️⃣  Testing Health Check...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Make sure app.py is running first!")
    exit()

# Test 2: Get Genres
print("\n2️⃣  Getting Available Genres...")
response = requests.get(f"{API_URL}/genres")
genres_data = response.json()
print(f"   Total genres: {genres_data['total_genres']}")
print(f"   Genres: {', '.join(genres_data['genres'][:5])}...")

# Test 3: Predict Revenue - RPG Game
print("\n3️⃣  Predicting RPG Game Revenue...")
test_game = {
    "genre": "rpg",
    "final_price": 480,
    "has_dlc": True,
    "dlc_available": 3,
    "win_support": True,
    "multi_platform": False
}

start_time = time.time()
response = requests.post(f"{API_URL}/predict", json=test_game)
latency_ms = (time.time() - start_time) * 1000

if response.status_code == 200:
    result = response.json()
    print(f"   ✅ Prediction successful!")
    print(f"   Genre: {result['genre'].upper()}")
    print(f"   Predicted Revenue: {result['predicted_revenue_formatted']}")
    print(f"   API Latency: {latency_ms:.2f}ms")
else:
    print(f"   ❌ Error: {response.json()}")

# Test 4: Compare Scenarios
print("\n4️⃣  Comparing Discount Impact...")
scenarios = [
    {"name": "Full Price", "genre": "action", "final_price": 500, "discount_pct_clean": 0},
    {"name": "50% Off", "genre": "action", "final_price": 500, "discount_pct_clean": 50, "is_on_sale": True}
]

for scenario in scenarios:
    response = requests.post(f"{API_URL}/predict", json=scenario)
    if response.status_code == 200:
        result = response.json()
        print(f"   {scenario['name']:<15} → {result['predicted_revenue_formatted']}")

print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED!")
print("=" * 70)
