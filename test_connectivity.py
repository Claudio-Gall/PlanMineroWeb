import requests
import os

# Load API Key
try:
    import streamlit as st
    api_key = st.secrets.get("GEMINI_API_KEY")
except:
    api_key = os.environ.get("GEMINI_API_KEY")

# Candidate from user's image (Plan C)
version = "v1beta"
model = "models/gemini-2.5-flash-lite-preview-02-05" 

url = f"https://generativelanguage.googleapis.com/{version}/{model}:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}
data = {"contents": [{"parts": [{"text": "Ping"}]}]}

print(f"📡 Testing Backup: {model}")
try:
    resp = requests.post(url, headers=headers, json=data, timeout=8)
    if resp.status_code == 200:
        print(f"✅ SUCCESS! Backup {model} is OPEN.")
    else:
        print(f"❌ Error {resp.status_code}: {resp.text[:200]}")
except Exception as e:
    print(f"❌ Exception: {e}")
