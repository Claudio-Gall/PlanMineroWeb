import requests
import urllib3
import json
urllib3.disable_warnings()

api_key = "AIzaSyCvEmGx0XohGmCvLCMhRHEqelJVxbd_C4s"

print("=" * 70)
print("TESTING GEMINI API - FINDING AVAILABLE MODELS")
print("=" * 70)

# Test 1: List models with v1
print("\n1. Testing v1 endpoint...")
url_v1 = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
try:
    s = requests.Session()
    s.trust_env = False
    resp = s.get(url_v1, verify=False, timeout=10)
    if resp.status_code == 200:
        models = resp.json().get('models', [])
        print(f"✅ Found {len(models)} models in v1:")
        for m in models[:5]:
            print(f"   - {m.get('name', 'N/A')}")
    else:
        print(f"❌ v1 error: {resp.status_code}")
except Exception as e:
    print(f"❌ v1 exception: {e}")

# Test 2: Try generating content with different model names
models_to_test = [
    ("v1", "gemini-pro"),
    ("v1", "models/gemini-pro"),
    ("v1beta", "gemini-pro"),
]

print("\n2. Testing generateContent endpoint...")
for api_version, model_name in models_to_test:
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
    data = {"contents": [{"parts": [{"text": "Hello"}]}]}
    
    try:
        s = requests.Session()
        s.trust_env = False
        resp = s.post(url, headers={"Content-Type": "application/json"}, json=data, verify=False, timeout=10)
        if resp.status_code == 200:
            print(f"✅ {api_version}/{model_name} works!")
            break
        else:
            print(f"❌ {api_version}/{model_name} error {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"❌ {api_version}/{model_name} exception: {str(e)[:100]}")

print("\n" + "=" * 70)
