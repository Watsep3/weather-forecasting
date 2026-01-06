"""
Script de déploiement automatique du meilleur modèle
Réutilise l'endpoint existant et crée un nouveau déploiement
"""

import os
import sys
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings,
    OnlineRequestSettings
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import time
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()


class ModelDeployer:
    """Classe pour gérer le déploiement des modèles"""
    
    def __init__(self):
        """Initialiser le client Azure ML"""
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
        
        if not all([self.subscription_id, self.resource_group, self.workspace_name]):
            raise ValueError("Missing Azure configuration. Check your .env file or GitHub secrets.")
        
        print("🔗 Connecting to Azure ML Workspace...")
        print(f"  Subscription: {self.subscription_id}")
        print(f"  Resource Group: {self.resource_group}")
        print(f"  Workspace: {self.workspace_name}")
        
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            self.subscription_id,
            self.resource_group,
            self.workspace_name
        )
        
        print("✅ Connected to Azure ML")
        
        # ✅ Nom d'endpoint FIXE pour réutilisation
        self.endpoint_name = "weather-api-prod"
        # ✅ Nom de déploiement avec timestamp pour versioning
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.deployment_name = f"weather-v{timestamp}"
        
        print(f"\n📍 Endpoint name (reusable): {self.endpoint_name}")
        print(f"📦 Deployment name (new): {self.deployment_name}")
    
    def check_available_models(self):
        """Vérifier quels modèles sont disponibles dans le registre"""
        print("\n📋 Checking available models in registry...")
        
        available_models = {
            'unified': False,
            'temperature': False,
            'rain': False
        }
        
        try:
            # Chercher le pipeline unifié
            try:
                unified_model = self.ml_client.models.get(
                    name="weather_unified_pipeline",
                    label="latest"
                )
                available_models['unified'] = True
                print("  ✅ Unified pipeline found")
                return 'unified', unified_model
            except Exception:
                print("  ⚠️ Unified pipeline not found")
            
            # Chercher le modèle de température
            try:
                temp_model_names = [
                    "weather_temperature_randomforest",
                    "weather_temperature_gradientboosting",
                    "weather_temperature_ridge",
                    "weather_temperature_lasso",
                    "weather_temperature_decisiontree"
                ]
                
                temp_model = None
                for name in temp_model_names:
                    try:
                        temp_model = self.ml_client.models.get(name=name, label="latest")
                        available_models['temperature'] = True
                        print(f"  ✅ Temperature model found: {name}")
                        break
                    except Exception:
                        continue
                
                if temp_model:
                    return 'temperature', temp_model
                else:
                    print("  ⚠️ Temperature model not found")
                    
            except Exception as e:
                print(f"  ⚠️ Error checking temperature model: {e}")
            
            print("\n❌ No models available for deployment")
            return None, None
            
        except Exception as e:
            print(f"\n❌ Error checking models: {e}")
            return None, None
    
    def create_or_update_endpoint(self):
        """Créer ou réutiliser l'endpoint existant"""
        print(f"\n🔧 Checking endpoint: {self.endpoint_name}")
        
        try:
            try:
                endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
                print(f"  ✅ Endpoint exists, will reuse it: {endpoint.name}")
                print(f"  📍 Endpoint URI: {endpoint.scoring_uri}")
                return endpoint
            except Exception:
                print("  📝 Endpoint doesn't exist, creating new one...")
            
            endpoint = ManagedOnlineEndpoint(
                name=self.endpoint_name,
                description="Weather forecasting API - temperature and rain prediction",
                auth_mode="key",
                tags={
                    "project": "weather-forecasting",
                    "type": "ml-inference",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": "unified-pipeline"
                }
            )
            
            print("  ⏳ Creating endpoint (2-3 minutes)...")
            endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            print(f"  ✅ Endpoint created: {endpoint.name}")
            
            return endpoint
            
        except Exception as e:
            print(f"  ❌ Error with endpoint: {e}")
            raise
    
    def list_existing_deployments(self):
        """Lister les déploiements existants sur l'endpoint"""
        try:
            deployments = self.ml_client.online_deployments.list(
                endpoint_name=self.endpoint_name
            )
            deployment_list = list(deployments)
            
            if deployment_list:
                print(f"\n📋 Existing deployments on {self.endpoint_name}:")
                for dep in deployment_list:
                    print(f"  • {dep.name}")
                return deployment_list
            else:
                print(f"\n📋 No existing deployments on {self.endpoint_name}")
                return []
        except Exception as e:
            print(f"  ⚠️ Could not list deployments: {e}")
            return []
    
    def deploy_model(self, model_type, model):
        """Déployer le modèle sur l'endpoint"""
        print(f"\n🚀 Deploying {model_type} model...")
        
        try:
            # ✅ Building environment from conda
            print("  🐳 Building environment from conda_env.yml...")
            print("  ⏰ This will take 10-15 minutes (building custom image)...")
            
            env = Environment(
                name="weather-forecast-env",
                conda_file="conda_env.yml",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
            )
            
            # Configuration du code
            code_config = CodeConfiguration(
                code=".",
                scoring_script="score.py"
            )
            
            # Configuration des timeouts
            request_settings = OnlineRequestSettings(
                request_timeout_ms=90000,  # 90 secondes
                max_concurrent_requests_per_instance=1
            )
            
            # Configuration du déploiement
            deployment = ManagedOnlineDeployment(
                name=self.deployment_name,
                endpoint_name=self.endpoint_name,
                model=model,
                environment=env,
                code_configuration=code_config,
                instance_type="Standard_F2s_v2",  # ✅ 2 cores seulement (économe)
                instance_count=1,
                request_settings=request_settings,
                liveness_probe=ProbeSettings(
                    failure_threshold=30,
                    success_threshold=1,
                    timeout=10,
                    period=10,
                    initial_delay=600  # 10 minutes pour le build
                ),
                readiness_probe=ProbeSettings(
                    failure_threshold=30,
                    success_threshold=1,
                    timeout=10,
                    period=10,
                    initial_delay=600  # 10 minutes
                ),
                tags={
                    "model_type": model_type,
                    "deployment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": self.deployment_name
                }
            )
            
            print("  📦 Creating deployment (10-15 minutes - building environment)...")
            print("  ⏰ Please be patient, environment is being built...")
            deployment = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
            
            print("  ✅ Deployment created successfully")
            
            # Allouer 100% du trafic à ce nouveau déploiement
            print("  🔀 Allocating traffic to new deployment...")
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            endpoint.traffic = {self.deployment_name: 100}
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            
            print("  ✅ Traffic allocated (100%)")
            
            return deployment
            
        except Exception as e:
            print(f"  ❌ Error deploying model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def delete_old_deployments(self):
        """Supprimer les anciens déploiements (garder seulement le nouveau)"""
        try:
            print(f"\n🧹 Cleaning up old deployments...")
            
            deployments = list(self.ml_client.online_deployments.list(
                endpoint_name=self.endpoint_name
            ))
            
            # Garder seulement le déploiement actuel
            old_deployments = [d for d in deployments if d.name != self.deployment_name]
            
            if not old_deployments:
                print("  ✅ No old deployments to clean up")
                return
            
            for deployment in old_deployments:
                print(f"  🗑️ Deleting old deployment: {deployment.name}")
                try:
                    self.ml_client.online_deployments.begin_delete(
                        name=deployment.name,
                        endpoint_name=self.endpoint_name
                    ).result()
                    print(f"  ✅ Deleted: {deployment.name}")
                except Exception as e:
                    print(f"  ⚠️ Could not delete {deployment.name}: {e}")
            
        except Exception as e:
            print(f"  ⚠️ Error during cleanup: {e}")
    
    def get_endpoint_info(self):
        """Récupérer les informations de l'endpoint"""
        try:
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            keys = self.ml_client.online_endpoints.get_keys(self.endpoint_name)
            
            print("\n" + "="*80)
            print("✅ DEPLOYMENT SUCCESSFUL")
            print("="*80)
            
            print(f"\n📍 Endpoint Information:")
            print(f"  Name: {endpoint.name}")
            print(f"  URI: {endpoint.scoring_uri}")
            print(f"  Status: {endpoint.provisioning_state}")
            print(f"  Location: {endpoint.location}")
            
            print(f"\n🔑 Authentication:")
            print(f"  Primary Key: {keys.primary_key[:30]}...")
            print(f"  Secondary Key: {keys.secondary_key[:30]}...")
            
            print(f"\n📊 Usage Example (Python):")
            print(f"""
import requests
import json

url = "{endpoint.scoring_uri}"
headers = {{
    "Content-Type": "application/json",
    "Authorization": f"Bearer {keys.primary_key}"
}}

data = {{
    "data": [{{
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
    }}]
}}

response = requests.post(url, headers=headers, json=data)
print(response.json())
            """)
            
            # Sauvegarder les infos dans un fichier
            output_file = f"endpoint_info_{self.endpoint_name}.txt"
            with open(output_file, "w") as f:
                f.write(f"Endpoint Name: {endpoint.name}\n")
                f.write(f"Endpoint URI: {endpoint.scoring_uri}\n")
                f.write(f"Primary Key: {keys.primary_key}\n")
                f.write(f"Secondary Key: {keys.secondary_key}\n")
                f.write(f"Status: {endpoint.provisioning_state}\n")
                f.write(f"Location: {endpoint.location}\n")
                f.write(f"Current Deployment: {self.deployment_name}\n")
                f.write(f"\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"\n💾 Endpoint information saved to: {output_file}")
            
            return endpoint, keys
            
        except Exception as e:
            print(f"❌ Error getting endpoint info: {e}")
            return None, None
    
    def run(self):
        """Exécuter le processus complet de déploiement"""
        print("\n" + "="*80)
        print("🌤️  WEATHER FORECASTING - AUTOMATED DEPLOYMENT")
        print("="*80)
        
        try:
            # 1. Vérifier les modèles disponibles
            model_type, model = self.check_available_models()
            
            if not model:
                print("\n❌ No models available for deployment")
                print("Please train your models first using train_pipeline.py")
                sys.exit(1)
            
            # 2. Créer ou réutiliser l'endpoint
            endpoint = self.create_or_update_endpoint()
            
            # 3. Lister les déploiements existants
            self.list_existing_deployments()
            
            # 4. Déployer le nouveau modèle
            deployment = self.deploy_model(model_type, model)
            
            # 5. Nettoyer les anciens déploiements
            self.delete_old_deployments()
            
            # 6. Récupérer les informations de l'endpoint
            self.get_endpoint_info()
            
            print("\n" + "="*80)
            print("✅ Deployment completed successfully!")
            print("="*80)
            print(f"\n🎯 Your API is ready!")
            print(f"📝 Endpoint: {self.endpoint_name} (reusable)")
            print(f"📦 Deployment: {self.deployment_name} (new version)")
            print(f"\n💡 Next deployments will reuse the same endpoint!")
            print(f"   Just run 'python deploy_model.py' again to update.")
            
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Point d'entrée du script"""
    deployer = ModelDeployer()
    deployer.run()


if __name__ == "__main__":
    main()