import requests
import json
import time
import os
from dotenv import load_dotenv

# ✅ NOUVELLES INFORMATIONS (depuis votre déploiement du 06/01/2026)
import os
from dotenv import load_dotenv

if not url or not api_key:
    raise ValueError("⚠️ Variables d'environnement manquantes. Vérifiez votre fichier .env")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "data": [{
        "hour": 14,
        "day_of_week": 1,
        "month": 1,
        "is_weekend": 0,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "city_encoded": 0,
        "condition_encoded": 1,
        "is_day": 1,
        "wind_kph": 15.0,
        "wind_degree": 180,
        "pressure_mb": 1013,
        "humidity": 65,
        "cloud_cover": 50,
        "uv_index": 5,
        "vis_km": 10,
        "temp_humidity_interaction": 9.75,
        "wind_temp_interaction": 225.0,
        "temp_lag_1": 18.0,
        "temp_lag_2": 17.5,
        "temp_lag_3": 17.0,
        "precip_lag_1": 0.0,
        "precip_lag_2": 0.0,
        "precip_lag_3": 0.0
    }]
}

print("🔄 Test de l'API Azure ML Weather Forecasting")
print(f"🌐 URL: {url}")
print(f"📦 Deployment: weather-v20260106-011952")
print("="*70)

max_retries = 5
retry_delay = 30

for attempt in range(1, max_retries + 1):
    print(f"\n📡 Tentative {attempt}/{max_retries}...")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*70)
            print("🎉 SUCCÈS ! L'API fonctionne correctement")
            print("="*70)
            print("\n📊 Réponse de l'API:")
            print(json.dumps(result, indent=2))
            print("\n✅ Test réussi !")
            break
            
        elif response.status_code in [424, 502, 503]:
            print(f"⏳ Service en cours d'initialisation (code {response.status_code})")
            print(f"💡 Le modèle met environ 5-10 minutes à démarrer après un déploiement")
            
            if attempt < max_retries:
                print(f"⏰ Attente de {retry_delay} secondes avant nouvelle tentative...")
                time.sleep(retry_delay)
            else:
                print("\n❌ Le service n'est toujours pas prêt après 5 tentatives")
                print("💡 Suggestions:")
                print("   - Vérifiez le statut dans le portail Azure")
                print("   - Le premier démarrage peut prendre jusqu'à 15 minutes")
                
        else:
            print(f"\n❌ Erreur HTTP {response.status_code}")
            print("Réponse:", response.text)
            break
            
    except requests.exceptions.Timeout:
        print("⏱️ Timeout - L'API met trop de temps à répondre")
        if attempt < max_retries:
            print(f"⏰ Nouvelle tentative dans {retry_delay} secondes...")
            time.sleep(retry_delay)
        
    except requests.exceptions.ConnectionError as e:
        print("❌ Erreur de connexion au serveur")
        if attempt < max_retries:
            print(f"⏰ Nouvelle tentative dans {retry_delay} secondes...")
            time.sleep(retry_delay)
        else:
            print(f"\nDétails: {str(e)[:200]}")
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {type(e).__name__}")
        print(f"Détails: {str(e)[:200]}")
        break

print("\n" + "="*70)
print("Test terminé")
print("="*70)