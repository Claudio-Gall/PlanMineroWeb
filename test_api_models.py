import requests
import os

# Load API key manually if needed, or from env
try:
    import streamlit as st
    api_key = st.secrets.get("GEMINI_API_KEY")
except:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    # Try to read from secrets.toml manually
    try:
        with open(".streamlit/secrets.toml", "r") as f:
            for line in f:
                if "GEMINI_API_KEY" in line:
                    api_key = line.split("=")[1].strip().strip('"')
                    break
    except:
        pass

if not api_key:
    print("❌ No API Key found.")
    exit()

print(f"🔑 Key found: {api_key[:5]}...")

def test_endpoint(version):
    url = f"https://generativelanguage.googleapis.com/{version}/models?key={api_key}"
    print(f"\n📡 Testing Endpoint: {version}")
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            print(f"✅ Success! Found {len(models)} models.")
            for m in models:
                if 'flash' in m['name'] or 'pro' in m['name']:
                    print(f"   - {m['name']}")
        else:
            print(f"❌ Error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"❌ Exception: {e}")

test_endpoint("v1")
test_endpoint("v1beta")
