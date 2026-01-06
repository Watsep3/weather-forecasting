"""
Script pour vérifier l'utilisation des cores Azure ML
"""

import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def check_cores_usage():
    """Vérifier quels endpoints utilisent des cores"""
    
    # Connexion à Azure ML
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
    
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name
    )
    
    print("=" * 80)
    print("🔍 AZURE ML - CORES USAGE ANALYSIS")
    print("=" * 80)
    
    # 1. Lister tous les endpoints
    print("\n📍 ONLINE ENDPOINTS:")
    print("-" * 80)
    
    endpoints = ml_client.online_endpoints.list()
    total_cores = 0
    endpoint_details = []
    
    for endpoint in endpoints:
        print(f"\n🌐 Endpoint: {endpoint.name}")
        print(f"   Status: {endpoint.provisioning_state}")
        
        # Gérer les endpoints en cours de suppression
        if endpoint.creation_context and endpoint.creation_context.created_at:
            print(f"   Created: {endpoint.creation_context.created_at}")
        else:
            print(f"   Created: N/A")
        
        # Skip les endpoints en cours de suppression
        if endpoint.provisioning_state == "Deleting":
            print(f"   ⏳ Endpoint is being deleted, skipping...")
            continue
        
        # Lister les déploiements de cet endpoint
        try:
            deployments = ml_client.online_deployments.list(endpoint_name=endpoint.name)
            
            endpoint_cores = 0
            for deployment in deployments:
                # Extraire le nombre de cores depuis l'instance type
                instance_type = deployment.instance_type
                instance_count = deployment.instance_count
                
                # Mapping des types d'instances vers les cores
                cores_map = {
                    "Standard_DS1_v2": 1,
                    "Standard_DS2_v2": 2,
                    "Standard_DS3_v2": 4,
                    "Standard_DS4_v2": 8,
                    "Standard_DS5_v2": 16,
                    "Standard_F2s_v2": 2,
                    "Standard_F4s_v2": 4,
                    "Standard_F8s_v2": 8,
                }
                
                cores = cores_map.get(instance_type, 0)
                deployment_cores = cores * instance_count
                endpoint_cores += deployment_cores
                
                print(f"   └── Deployment: {deployment.name}")
                print(f"       Instance: {instance_type} (x{instance_count})")
                print(f"       Cores: {cores} × {instance_count} = {deployment_cores} cores")
                print(f"       Status: {deployment.provisioning_state}")
                
                endpoint_details.append({
                    "endpoint": endpoint.name,
                    "deployment": deployment.name,
                    "instance_type": instance_type,
                    "instance_count": instance_count,
                    "cores": deployment_cores,
                    "status": deployment.provisioning_state
                })
            
            total_cores += endpoint_cores
            print(f"   ➡️  Total endpoint cores: {endpoint_cores}")
            
        except Exception as e:
            print(f"   ⚠️  Error listing deployments: {e}")
    
    # 2. Résumé
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"\n🔢 Total cores in use: {total_cores}")
    print(f"📈 Quota limit: 6 cores (France Central)")
    print(f"📉 Available: {6 - total_cores} cores")
    
    if total_cores > 6:
        print(f"\n⚠️  WARNING: You're using {total_cores - 6} cores over your quota!")
    
    # 3. Recommandations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    if len(endpoint_details) > 1:
        print("\n🗑️  Endpoints to delete (keeping only the latest):")
        # Trier par date de création (garder le plus récent)
        endpoint_details_sorted = sorted(
            endpoint_details, 
            key=lambda x: x['endpoint'], 
            reverse=True
        )
        
        for i, detail in enumerate(endpoint_details_sorted[1:], 1):
            print(f"\n{i}. Delete: {detail['endpoint']}")
            print(f"   Will free: {detail['cores']} cores")
            print(f"   Command: az ml online-endpoint delete --name {detail['endpoint']} --yes")
    
    # 4. Commandes utiles
    print("\n" + "=" * 80)
    print("🛠️  USEFUL COMMANDS")
    print("=" * 80)
    print("\n# List all endpoints:")
    print("az ml online-endpoint list")
    print("\n# Delete an endpoint:")
    print("az ml online-endpoint delete --name <endpoint-name> --yes")
    print("\n# Check quota:")
    print("az ml workspace show --query compute")
    
    return endpoint_details, total_cores


if __name__ == "__main__":
    try:
        check_cores_usage()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()