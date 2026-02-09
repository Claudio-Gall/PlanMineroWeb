import requests
import time

URL = "https://budget2026.streamlit.app"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
}

def ping_app():
    print(f"Despertando a {URL}...")
    max_retries = 3
    for i in range(max_retries):
        try:
            print(f"Intento {i+1}/{max_retries}...")
            start_time = time.time()
            response = requests.get(URL, headers=HEADERS, timeout=60) # Increased to 60s
            elapsed = time.time() - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"Tiempo de respuesta: {elapsed:.2f} segundos")
            
            if response.status_code == 200:
                print("✅ ÉXITO: La aplicación está despierta.")
                return
            else:
                print(f"⚠️ ALERTA: Código de estado inesperado {response.status_code}")
                
        except requests.exceptions.Timeout:
             print(f"⏳ TIMEOUT: El intento {i+1} excedió los 60 segundos.")
        except Exception as e:
            print(f"❌ ERROR: Excepción en intento {i+1}: {e}")
        
        # Wait before retry
        if i < max_retries - 1:
            time.sleep(10)
            
    print("❌ FATAL: No se pudo despertar la aplicación tras varios intentos.")
    exit(1)

if __name__ == "__main__":
    ping_app()
