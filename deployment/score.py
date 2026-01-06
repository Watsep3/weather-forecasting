"""
Score Script pour Azure ML Endpoint
Gère les prédictions avec le pipeline unifié ou température seule
"""

import json
import logging
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    """
    Fonction appelée une seule fois au démarrage de l'endpoint
    Charge le modèle depuis le registre
    """
    global model
    global model_type
    global feature_names
    global scaler  # ⭐ AJOUT DU SCALER
    
    try:
        # Récupérer le chemin du modèle depuis la variable d'environnement
        model_path = os.getenv("AZUREML_MODEL_DIR")
        
        if not model_path:
            logger.error("AZUREML_MODEL_DIR not set")
            raise ValueError("Model directory not found")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Lister tous les fichiers pour debug
        logger.info("Directory structure:")
        for root, dirs, files in os.walk(model_path):
            logger.info(f"  Directory: {root}")
            for file in files:
                logger.info(f"    File: {file}")
        
        # Chemins possibles pour le pipeline
        possible_pipeline_paths = [
            os.path.join(model_path, "unified_pipeline", "model.pkl"),
            os.path.join(model_path, "models", "unified_pipeline.pkl"),
            os.path.join(model_path, "unified_pipeline.pkl"),
            os.path.join(model_path, "weather_unified_pipeline", "models", "unified_pipeline.pkl"),
            os.path.join(model_path, "weather_unified_pipeline", "unified_pipeline.pkl"),
        ]
        
        pipeline_path = None
        for path in possible_pipeline_paths:
            if os.path.exists(path):
                pipeline_path = path
                logger.info(f"✅ Found pipeline at: {path}")
                break
        
        if not pipeline_path:
            logger.error(f"Pipeline not found in any of: {possible_pipeline_paths}")
            raise FileNotFoundError(f"Pipeline not found in: {model_path}")
        
        # Charger le pipeline
        logger.info(f"Loading pipeline from: {pipeline_path}")
        with open(pipeline_path, 'rb') as f:
            model = pickle.load(f)
        
        model_type = "unified"
        logger.info("✅ Unified pipeline loaded successfully")
        
        # ⭐ CHARGER LE SCALER
        possible_scaler_paths = [
            os.path.join(model_path, "models", "scaler.pkl"),
            os.path.join(model_path, "scaler.pkl"),
            os.path.join(model_path, "weather_unified_pipeline", "models", "scaler.pkl"),
        ]
        
        scaler_path = None
        for path in possible_scaler_paths:
            if os.path.exists(path):
                scaler_path = path
                logger.info(f"✅ Found scaler at: {path}")
                break
        
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✅ Scaler loaded successfully")
        else:
            # Si pas de scaler trouvé, utiliser celui du pipeline
            if hasattr(model, 'scaler') and model.scaler is not None:
                scaler = model.scaler
                logger.info("✅ Using scaler from pipeline")
            else:
                logger.warning("⚠️ No scaler found - predictions may be incorrect!")
                scaler = None
        
        # Features attendues (24 features)
        feature_names = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'hour_sin', 'hour_cos',
            'city_encoded', 'condition_encoded',
            'is_day', 'wind_kph', 'wind_degree', 'pressure_mb',
            'humidity', 'cloud_cover', 'uv_index', 'vis_km',
            'temp_humidity_interaction', 'wind_temp_interaction',
            'temp_lag_1', 'temp_lag_2', 'temp_lag_3',
            'precip_lag_1', 'precip_lag_2', 'precip_lag_3'
        ]
        
        logger.info(f"Model type: {model_type}")
        logger.info(f"Expected features: {len(feature_names)}")
        logger.info(f"Scaler available: {scaler is not None}")
        logger.info("✅ Initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during initialization: {str(e)}", exc_info=True)
        raise


def run(raw_data: str) -> str:
    """
    Fonction appelée pour chaque prédiction
    
    Args:
        raw_data: JSON string contenant les données d'entrée
        
    Returns:
        JSON string avec les prédictions
    """
    try:
        logger.info("Received prediction request")
        
        # Parser les données d'entrée
        data = json.loads(raw_data)
        logger.info(f"Input data keys: {data.keys()}")
        
        # Vérifier le format des données
        if "data" not in data:
            return json.dumps({
                "success": False,
                "error": "Missing 'data' field in input",
                "example": {
                    "data": [{
                        "hour": 14,
                        "day_of_week": 1,
                        # ... (reste de l'exemple)
                    }]
                }
            })
        
        input_data = data["data"]
        
        # Convertir en DataFrame
        df = pd.DataFrame(input_data)
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Vérifier les features manquantes
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0
        
        # Réorganiser les colonnes dans le bon ordre
        X = df[feature_names].values
        logger.info(f"Input shape before scaling: {X.shape}")
        logger.info(f"Input sample (first row): {X[0]}")
        
        # ⭐ NORMALISER LES DONNÉES AVANT PRÉDICTION
        if scaler is not None:
            X_scaled = scaler.transform(X)
            logger.info(f"✅ Data scaled")
            logger.info(f"Scaled sample (first row): {X_scaled[0]}")
        else:
            logger.warning("⚠️ No scaler available - using raw data")
            X_scaled = X
        
        # Faire les prédictions
        if model_type == "unified":
            # Prédiction température avec données normalisées
            temp_pred = model.temp_model.predict(X_scaled)
            logger.info(f"Temperature predictions: {temp_pred}")
            
            # Vérifier si les températures sont réalistes
            if np.any(temp_pred < -50) or np.any(temp_pred > 60):
                logger.warning(f"⚠️ Unusual temperature detected: {temp_pred}")
            
            # Créer les features pour la prédiction de pluie
            X_rain = np.column_stack([X_scaled, temp_pred])
            
            # Prédiction pluie
            rain_pred = model.rain_model.predict(X_rain)
            logger.info(f"Rain predictions: {rain_pred}")
            
            # Probabilités de pluie
            try:
                rain_proba_all = model.rain_model.predict_proba(X_rain)
                if rain_proba_all.shape[1] == 2:
                    rain_proba = rain_proba_all[:, 1]
                else:
                    rain_proba = np.zeros(len(rain_pred))
            except Exception as proba_error:
                logger.warning(f"Could not get probabilities: {proba_error}")
                rain_proba = rain_pred.astype(float)
            
            # Construire les résultats
            results = []
            for i in range(len(temp_pred)):
                results.append({
                    "temperature_celsius": float(temp_pred[i]),
                    "will_rain": bool(rain_pred[i]),
                    "rain_probability": float(rain_proba[i]),
                    "confidence": "high" if abs(rain_proba[i] - 0.5) > 0.3 else "medium"
                })
            
            response = {
                "success": True,
                "model_type": "unified",
                "predictions": results,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        else:  # temperature_only
            temperature_predictions = model.predict(X_scaled)
            
            results = []
            for i, temp in enumerate(temperature_predictions):
                results.append({
                    "temperature_celsius": float(temp),
                    "rain_prediction": "unavailable",
                    "message": "Rain model not trained due to insufficient data"
                })
            
            response = {
                "success": True,
                "model_type": "temperature_only",
                "predictions": results,
                "timestamp": pd.Timestamp.now().isoformat(),
                "warning": "Rain predictions unavailable"
            }
        
        logger.info("✅ Predictions completed successfully")
        return json.dumps(response)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return json.dumps({
            "success": False,
            "error": "Invalid JSON format",
            "details": str(e)
        })
        
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
            "model_type": model_type if 'model_type' in globals() else "unknown"
        })