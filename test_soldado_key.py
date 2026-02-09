import toml
import requests

def test_key():
    print("🔑 Verificando Nueva Llave API (Soldado)...")
    
    # 1. Load Secrets
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        api_key = secrets.get("GEMINI_API_KEY")
    except Exception as e:
        print(f"❌ Error leyendo secrets.toml: {e}")
        return

    if not api_key:
        print("❌ No encontré GEMINI_API_KEY en secrets.toml")
        return

    # 2. Test Gemini API (Gemini Flash Latest)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": "Write a python function to sum two numbers."}]}]}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            print("\n✅ ¡ÉXITO! La llave funciona con Gemini 2.0 Flash.")
            print(f"🤖 Respuesta: {response.json()['candidates'][0]['content']['parts'][0]['text'][:100]}...")
            print("\n🎉 Proyecto 'soldado-f982f' configurado correctamente.")
        else:
            print(f"\n❌ Error API ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"\n❌ Error de Conexión: {e}")

if __name__ == "__main__":
    test_key()
