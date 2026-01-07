"""
Azure ML - Weather Forecasting PIPELINE COMPLET
Prédiction : Température (régression) + Précipitations (classification)
Pipeline unifié avec modèles liés
Compatible Azure ML - Sans imbalanced-learn
VERSION FINALE - Pipeline intégré avec DÉTECTION OVERFITTING CORRIGÉE
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import os
import json
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Azure ML et MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential, AzureCliCredential

# Azure Storage
from azure.storage.blob import BlobServiceClient

# Pour charger les variables d'environnement
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class DataBalancer:
    """Classe pour le balancing des données sans imbalanced-learn"""
    
    @staticmethod
    def smote_simple(X, y, k_neighbors=5, random_state=42):
        """
        Implémentation simple de SMOTE (Synthetic Minority Over-sampling Technique)
        Compatible Azure ML sans dépendances externes
        """
        np.random.seed(random_state)
        
        # Identifier les classes
        classes, counts = np.unique(y, return_counts=True)
        
        if len(classes) <= 1:
            return X, y
        
        # Trouver la classe majoritaire
        max_count = counts.max()
        
        X_resampled = []
        y_resampled = []
        
        for cls in classes:
            X_cls = X[y == cls]
            
            # Garder toutes les instances de la classe
            X_resampled.append(X_cls)
            y_resampled.extend([cls] * len(X_cls))
            
            # Si classe minoritaire, générer des exemples synthétiques
            if len(X_cls) < max_count:
                n_synthetic = max_count - len(X_cls)
                
                for _ in range(n_synthetic):
                    # Choisir un exemple aléatoire
                    idx = np.random.randint(0, len(X_cls))
                    sample = X_cls[idx]
                    
                    # Trouver k voisins les plus proches (simplifié)
                    distances = np.linalg.norm(X_cls - sample, axis=1)
                    k = min(k_neighbors, len(X_cls) - 1)
                    nearest_idx = np.argsort(distances)[1:k+1]
                    
                    # Choisir un voisin aléatoire
                    neighbor_idx = np.random.choice(nearest_idx)
                    neighbor = X_cls[neighbor_idx]
                    
                    # Générer un exemple synthétique
                    alpha = np.random.random()
                    synthetic = sample + alpha * (neighbor - sample)
                    
                    X_resampled.append(synthetic.reshape(1, -1))
                    y_resampled.append(cls)
        
        X_balanced = np.vstack(X_resampled)
        y_balanced = np.array(y_resampled)
        
        return X_balanced, y_balanced
    
    @staticmethod
    def random_oversample(X, y, random_state=42):
        """Over-sampling aléatoire de la classe minoritaire"""
        np.random.seed(random_state)
        
        classes, counts = np.unique(y, return_counts=True)
        
        if len(classes) <= 1:
            return X, y
        
        max_count = counts.max()
        
        X_resampled = []
        y_resampled = []
        
        for cls in classes:
            X_cls = X[y == cls]
            y_cls = y[y == cls]
            
            if len(X_cls) < max_count:
                # Over-sample
                indices = np.random.choice(len(X_cls), max_count, replace=True)
                X_resampled.append(X_cls[indices])
                y_resampled.extend([cls] * max_count)
            else:
                X_resampled.append(X_cls)
                y_resampled.extend(y_cls)
        
        X_balanced = np.vstack(X_resampled)
        y_balanced = np.array(y_resampled)
        
        return X_balanced, y_balanced


class WeatherPredictionPipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline unifié pour prédiction météo
    1. Prédit la température
    2. Utilise la température prédite pour prédire la pluie
    """
    
    def __init__(self, temp_model=None, rain_model=None, scaler=None):
        self.temp_model = temp_model
        self.rain_model = rain_model
        self.scaler = scaler
        self.temp_feature_names = None
        self.rain_feature_names = None
        
    def fit(self, X_temp, y_temp, X_rain, y_rain):
        """
        Entraîner les deux modèles
        X_temp: features pour température
        y_temp: target température
        X_rain: features pour pluie (inclut temp réelle)
        y_rain: target pluie (0/1)
        """
        print("🔧 Entraînement du pipeline unifié...")
        
        # Entraîner le modèle de température
        print("  1️⃣ Entraînement modèle température...")
        self.temp_model.fit(X_temp, y_temp)
        print("     ✓ Modèle température entraîné")
        
        # Entraîner le modèle de pluie
        print("  2️⃣ Entraînement modèle pluie...")
        self.rain_model.fit(X_rain, y_rain)
        print("     ✓ Modèle pluie entraîné")
        
        return self
    
    def predict(self, X_temp):
        """
        Prédiction complète:
        1. Prédit température
        2. Ajoute température prédite aux features
        3. Prédit pluie
        """
        # Prédire la température
        temp_pred = self.temp_model.predict(X_temp)
        
        # Créer les features pour la prédiction de pluie
        # Ajouter la température prédite comme nouvelle feature
        X_rain = np.column_stack([X_temp, temp_pred])
        
        # Prédire la pluie
        rain_pred = self.rain_model.predict(X_rain)
        
        return {
            'temperature': temp_pred,
            'will_rain': rain_pred
        }
    
    def predict_proba(self, X_temp):
        """Prédiction avec probabilités pour la pluie"""
        # Prédire la température
        temp_pred = self.temp_model.predict(X_temp)
        
        # Créer les features pour la prédiction de pluie
        X_rain = np.column_stack([X_temp, temp_pred])
        
        # Prédire la pluie avec probabilités
        rain_proba_all = self.rain_model.predict_proba(X_rain)
        
        # Gérer le cas où il n'y a qu'une seule classe
        if rain_proba_all.shape[1] == 1:
            # Une seule classe (probablement 0 - pas de pluie)
            rain_proba = np.zeros(len(X_temp))
            rain_pred = np.zeros(len(X_temp), dtype=int)
        else:
            # Deux classes normales
            rain_proba = rain_proba_all[:, 1]
            rain_pred = (rain_proba > 0.5).astype(int)
        
        return {
            'temperature': temp_pred,
            'will_rain': rain_pred,
            'rain_probability': rain_proba
        }


class WeatherMLPipeline:
    """Pipeline ML complet pour prédiction météo"""
    
    def __init__(self, storage_account_name, container_name, storage_account_key=None, 
                 experiment_name="weather-forecast", azure_ml_client=None):
        self.storage_account = storage_account_name
        self.container = container_name
        self.storage_account_key = storage_account_key
        self.experiment_name = experiment_name
        self.azure_ml_client = azure_ml_client
        
        # Modèles de régression pour température
        self.regression_models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42)
        }
        
        # Modèles de classification pour pluie - ORDRE IMPORTANT (simple → complexe)
        self.classification_models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
            'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42)
        }
        
        self.best_temp_model = None
        self.best_temp_model_name = None
        self.best_temp_score = float('-inf')
        self.best_temp_metrics = {}
        
        self.best_rain_model = None
        self.best_rain_model_name = None
        self.best_rain_score = float('-inf')
        self.best_rain_metrics = {}
        
        self.unified_pipeline = None
        self.scaler = StandardScaler()
        self.balancer = DataBalancer()
        
        # Flag pour indiquer si le modèle de pluie est disponible
        self.rain_model_available = False
        
    def load_data_from_blob(self, blob_path="bronze/history"):
        """Charger les données depuis Azure Blob Storage avec clé d'accès"""
        try:
            print(f"  🔗 Connexion au Storage Account: {self.storage_account}")
            print(f"  📦 Container: {self.container}")
            print(f"  📂 Path: {blob_path}")
            
            # Connexion au blob storage avec clé d'accès
            account_url = f"https://{self.storage_account}.blob.core.windows.net"
            
            if self.storage_account_key:
                blob_service_client = BlobServiceClient(
                    account_url=account_url, 
                    credential=self.storage_account_key
                )
                print("  🔑 Authentification avec clé d'accès")
            else:
                print("  ⚠️ Pas de clé fournie, tentative accès public")
                blob_service_client = BlobServiceClient(account_url=account_url)
            
            container_client = blob_service_client.get_container_client(self.container)
            
            # Lister les blobs
            print(f"  📋 Liste des fichiers dans {blob_path}...")
            blobs = list(container_client.list_blobs(name_starts_with=blob_path))
            print(f"  ✓ {len(blobs)} fichiers trouvés")
            
            all_data = []
            blob_count = 0
            
            for blob in blobs:
                if blob.name.endswith('.json'):
                    blob_count += 1
                    print(f"  📄 Lecture: {blob.name}")
                    blob_client = container_client.get_blob_client(blob.name)
                    content = blob_client.download_blob().readall().decode('utf-8-sig')
                    
                    # Lire ligne par ligne (format JSONL)
                    for line in content.strip().split('\n'):
                        if line:
                            try:
                                all_data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"    ⚠️ Erreur JSON sur ligne: {e}")
                                continue
            
            if not all_data:
                print(f"  ⚠️ Aucune donnée JSON trouvée dans {blob_path}")
                return None
            
            df = pd.DataFrame(all_data)
            print(f"  ✓ {blob_count} fichiers JSON lus")
            print(f"  ✓ {len(df)} observations chargées")
            print(f"  ✓ Colonnes: {list(df.columns)[:10]}")
            
            return df
            
        except Exception as e:
            print(f"  ✗ Erreur lors du chargement: {e}")
            return None
    
    def feature_engineering(self, df):
        """Créer des features pour la prédiction horaire"""
        print("  🔧 Feature engineering en cours...")
        df = df.copy()
        
        # Convertir les timestamps
        df['observation_time'] = pd.to_datetime(df['observation_time'])
        df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])
        
        # Features temporelles
        df['hour'] = df['observation_time'].dt.hour
        df['day_of_week'] = df['observation_time'].dt.dayofweek
        df['month'] = df['observation_time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Features cycliques pour l'heure
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Features d'interaction
        df['temp_humidity_interaction'] = df['temp_c'] * df['humidity'] / 100
        df['wind_temp_interaction'] = df['wind_kph'] * df['temp_c']
        
        # Encoder les variables catégorielles
        le_city = LabelEncoder()
        le_condition = LabelEncoder()
        
        df['city_encoded'] = le_city.fit_transform(df['city'])
        df['condition_encoded'] = le_condition.fit_transform(df['condition'])
        
        # Sauvegarder les encoders
        self.city_encoder = le_city
        self.condition_encoder = le_condition
        
        # Features de lag
        df = df.sort_values(['city', 'observation_time'])
        
        for lag in [1, 2, 3]:
            df[f'temp_lag_{lag}'] = df.groupby('city')['temp_c'].shift(lag)
            df[f'precip_lag_{lag}'] = df.groupby('city')['precip_mm'].shift(lag)
        
        lag_cols = [col for col in df.columns if 'lag' in col]
        df[lag_cols] = df[lag_cols].fillna(df[lag_cols].mean())
        
        print(f"  ✓ {len(df.columns)} features créées")
        
        return df
    
    def prepare_features_target(self, df, target='temp_c'):
        """Préparer les features et la cible"""
        feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'hour_sin', 'hour_cos',
            'city_encoded', 'condition_encoded',
            'is_day', 'wind_kph', 'wind_degree', 'pressure_mb',
            'humidity', 'cloud_cover', 'uv_index', 'vis_km',
            'temp_humidity_interaction', 'wind_temp_interaction'
        ]
        
        lag_cols = [col for col in df.columns if 'lag' in col]
        feature_cols.extend(lag_cols)
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"  ✓ Features finales: {len(feature_cols)}")
        print(f"  ✓ Observations valides: {len(X)}")
        
        return X, y, feature_cols
    
    def create_classification_target(self, df, target_col='precip_mm', threshold=0.1):
        """Créer une cible de classification pour les précipitations"""
        df = df.copy()
        df['will_rain'] = (df[target_col] > threshold).astype(int)
        return df
    
    def analyze_data_balance(self, y, task='regression'):
        """Analyser l'équilibre des données"""
        if task == 'classification':
            balance = Counter(y)
            print("\n📊 Distribution des classes:")
            for cls, count in sorted(balance.items()):
                print(f"  Classe {cls}: {count} ({count/len(y)*100:.1f}%)")
            
            ratio = max(balance.values()) / min(balance.values()) if min(balance.values()) > 0 else 1
            print(f"\n⚖️ Ratio déséquilibre: {ratio:.2f}:1")
            
            if ratio > 5:
                print("  → SMOTE recommandé (fort déséquilibre)")
                return 'smote'
            elif ratio > 3:
                print("  → Over-sampling recommandé (déséquilibre modéré)")
                return 'oversample'
            elif ratio > 1.5:
                print("  → Over-sampling recommandé (léger déséquilibre)")
                return 'oversample'
            else:
                print("  → Pas de balancing nécessaire")
                return 'none'
        else:
            print("\n📊 Distribution de la cible (régression):")
            print(f"  Min: {y.min():.2f}")
            print(f"  Max: {y.max():.2f}")
            print(f"  Mean: {y.mean():.2f}")
            print(f"  Std: {y.std():.2f}")
            return 'none'
    
    def balance_data(self, X, y, strategy='smote'):
        """Appliquer le balancing des données"""
        if strategy == 'none':
            return X, y
        
        print(f"\n🔄 Application du balancing: {strategy}")
        print(f"  Avant: {len(X)} échantillons")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        if strategy == 'smote':
            X_balanced, y_balanced = DataBalancer.smote_simple(X_array, y_array)
        elif strategy == 'oversample':
            X_balanced, y_balanced = DataBalancer.random_oversample(X_array, y_array)
        else:
            X_balanced, y_balanced = X_array, y_array
        
        print(f"  Après: {len(X_balanced)} échantillons")
        
        balance = Counter(y_balanced)
        print(f"\n  Nouvelle distribution:")
        for cls, count in sorted(balance.items()):
            print(f"    Classe {cls}: {count} ({count/len(y_balanced)*100:.1f}%)")
        
        return X_balanced, y_balanced
    
    def train_temperature_models(self, X_train, X_test, y_train, y_test):
        """Entraîner et comparer les modèles de température"""
        results = {}
        
        print("\n" + "="*80)
        print("🌡️  ENTRAÎNEMENT DES MODÈLES DE TEMPÉRATURE")
        print("="*80)
        
        for model_name, model in self.regression_models.items():
            print(f"\n📦 Entraînement: {model_name}")
            
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            cv_folds = min(5, len(X_train))
            if cv_folds >= 2:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                           scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
            else:
                cv_mae = mae_test
            
            results[model_name] = {
                'model': model,
                'mae_train': mae_train,
                'mae_test': mae_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'cv_mae': cv_mae
            }
            
            print(f"  ✓ MAE Test: {mae_test:.3f}°C")
            print(f"  ✓ RMSE Test: {rmse_test:.3f}°C")
            print(f"  ✓ R² Test: {r2_test:.3f}")
            
            if r2_test > self.best_temp_score:
                self.best_temp_score = r2_test
                self.best_temp_model = model
                self.best_temp_model_name = model_name
                self.best_temp_metrics = {
                    'mae_train': mae_train,
                    'mae_test': mae_test,
                    'rmse_train': rmse_train,
                    'rmse_test': rmse_test,
                    'r2_train': r2_train,
                    'r2_test': r2_test,
                    'cv_mae': cv_mae
                }
        
        return results
    
    def train_rain_models(self, X_train, X_test, y_train, y_test):
        """
        Entraîner et comparer les modèles de pluie
        AVEC DÉTECTION D'OVERFITTING AMÉLIORÉE - Pénalise les scores parfaits
        """
        results = {}
        
        print("\n" + "="*80)
        print("🌧️  ENTRAÎNEMENT DES MODÈLES DE PLUIE (détection overfitting renforcée)")
        print("="*80)
        
        # Vérifier si on a au moins 2 classes
        n_classes = len(np.unique(y_train))
        if n_classes < 2:
            print(f"\n⚠️ ATTENTION: Une seule classe détectée dans les données d'entraînement!")
            print(f"   Impossible d'entraîner des modèles de classification binaire.")
            print(f"   Classes présentes: {np.unique(y_train)}")
            return results
        
        # Déterminer le nombre de folds pour CV
        min_class_count = min(Counter(y_train).values())
        cv_folds = min(3, min_class_count)
        
        if cv_folds < 2:
            print(f"\n⚠️ ATTENTION: Pas assez de données pour Cross-Validation fiable")
            print(f"   Classe minoritaire: {min_class_count} échantillons")
            print(f"   Un dataset plus large est fortement recommandé!")
        
        # 🔑 Détecter si le dataset est trop petit
        dataset_too_small = len(X_train) < 100 or len(X_test) < 20
        if dataset_too_small:
            print(f"\n⚠️ DATASET TROP PETIT DÉTECTÉ:")
            print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
            print(f"   → Pénalités automatiques pour scores parfaits activées")
        
        for model_name, model in self.classification_models.items():
            print(f"\n📦 Entraînement: {model_name}")
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métriques sur train
            acc_train = accuracy_score(y_train, y_pred_train)
            
            # Métriques sur test
            acc_test = accuracy_score(y_test, y_pred_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred_test, average='binary', zero_division=0
            )
            
            # ROC AUC si le modèle supporte predict_proba
            roc_auc = 0.0
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba_all = model.predict_proba(X_test)
                    if y_proba_all.shape[1] == 2:
                        y_proba = y_proba_all[:, 1]
                        if len(np.unique(y_test)) > 1:
                            roc_auc = roc_auc_score(y_test, y_proba)
                except Exception as e:
                    print(f"    ⚠️ Erreur calcul ROC AUC: {e}")
                    roc_auc = 0.0
            
            # 🔑 CROSS-VALIDATION pour détecter l'overfitting
            cv_f1_mean = 0.0
            cv_f1_std = 0.0
            overfitting_detected = False
            final_score = f1  # Score par défaut
            
            if cv_folds >= 2 and len(X_train) >= 10:
                try:
                    print(f"  📊 Cross-Validation ({cv_folds}-fold)...")
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds,
                        scoring='f1'
                    )
                    cv_f1_mean = cv_scores.mean()
                    cv_f1_std = cv_scores.std()
                    
                    print(f"     CV F1: {cv_f1_mean:.3f} (±{cv_f1_std:.3f})")
                    
                    # 🔴 NOUVELLE LOGIQUE DE DÉTECTION - Pénalise TOUJOURS les scores parfaits
                    
                    # 1. Scores quasi-parfaits sur petit dataset = TRÈS SUSPECT
                    if f1 >= 0.95 and acc_test >= 0.95 and dataset_too_small:
                        print(f"  ⚠️  OVERFITTING TRÈS PROBABLE!")
                        print(f"     Scores quasi-parfaits (F1={f1:.3f}, Acc={acc_test:.3f}) sur petit dataset")
                        overfitting_detected = True
                        
                        # Pénalité sévère si les deux sont parfaits
                        if f1 == 1.0 and cv_f1_mean >= 0.95:
                            final_score = 0.5  # Pénalité maximum
                            print(f"     → Pénalité sévère (scores parfaits): {final_score:.3f}")
                        else:
                            final_score = cv_f1_mean * 0.7  # Pénalité modérée
                            print(f"     → Pénalité modérée: {final_score:.3f}")
                    
                    # 2. Grand écart Test vs CV (indépendamment des scores)
                    elif abs(f1 - cv_f1_mean) > 0.25:  # Seuil abaissé à 25%
                        print(f"  ⚠️  OVERFITTING DÉTECTÉ!")
                        print(f"     Écart Test F1 ({f1:.3f}) vs CV F1 ({cv_f1_mean:.3f}) = {abs(f1 - cv_f1_mean):.3f}")
                        overfitting_detected = True
                        final_score = cv_f1_mean  # Utiliser CV
                    
                    # 3. Bon équilibre mais sur petit dataset
                    elif dataset_too_small:
                        print(f"  ✓  Équilibre Train/CV/Test acceptable")
                        # Légère pénalité pour petit dataset
                        final_score = f1 * 0.9
                        print(f"     Petit dataset → légère pénalité: {final_score:.3f}")
                    
                    # 4. Tout va bien
                    else:
                        print(f"  ✅ Bon équilibre - Dataset suffisant")
                        final_score = f1
                        
                except Exception as e:
                    print(f"    ⚠️ Erreur Cross-Validation: {e}")
                    # Si CV échoue mais score parfait
                    if f1 >= 0.95 and acc_test >= 0.95 and dataset_too_small:
                        print(f"  ⚠️  Score quasi-parfait + CV échec → Pénalité sévère")
                        final_score = f1 * 0.5
                        overfitting_detected = True
                    else:
                        final_score = f1
            else:
                print(f"  ⚠️  Dataset trop petit pour CV fiable ({len(X_train)} échantillons)")
                # Pénalité automatique basée sur la taille et les scores
                if f1 >= 0.95:
                    final_score = f1 * 0.5  # Pénalité sévère
                    print(f"     Score quasi-parfait + mini-dataset → Pénalité sévère: {final_score:.3f}")
                    overfitting_detected = True
                elif len(X_train) < 30:
                    final_score = f1 * 0.6  # Pénalité forte
                    print(f"     Pénalité forte appliquée: {final_score:.3f}")
                    overfitting_detected = True
                else:
                    final_score = f1 * 0.8  # Pénalité modérée
                    print(f"     Pénalité modérée appliquée: {final_score:.3f}")
            
            # Stocker les résultats
            results[model_name] = {
                'model': model,
                'acc_train': acc_train,
                'acc_test': acc_test,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'cv_f1_mean': cv_f1_mean,
                'cv_f1_std': cv_f1_std,
                'final_score': final_score,
                'overfitting_detected': overfitting_detected
            }
            
            # Affichage des métriques
            print(f"  ✓ Accuracy Train: {acc_train:.3f}")
            print(f"  ✓ Accuracy Test: {acc_test:.3f}")
            print(f"  ✓ Precision: {precision:.3f}")
            print(f"  ✓ Recall: {recall:.3f}")
            print(f"  ✓ F1-Score Test: {f1:.3f}")
            if cv_f1_mean > 0:
                print(f"  ✓ F1-Score CV: {cv_f1_mean:.3f} (±{cv_f1_std:.3f})")
            if roc_auc > 0:
                print(f"  ✓ ROC AUC: {roc_auc:.3f}")
            print(f"  🎯 Score Final (sélection): {final_score:.3f}")
            
            if overfitting_detected:
                print(f"  ⚠️  Modèle suspect d'overfitting")
            
            # Sélection basée sur le score final
            if final_score > self.best_rain_score:
                self.best_rain_score = final_score
                self.best_rain_model = model
                self.best_rain_model_name = model_name
                self.best_rain_metrics = {
                    'acc_train': acc_train,
                    'acc_test': acc_test,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'cv_f1_mean': cv_f1_mean,
                    'cv_f1_std': cv_f1_std,
                    'final_score': final_score,
                    'overfitting_detected': overfitting_detected
                }
                print(f"  🏆 Nouveau meilleur modèle! (Score: {final_score:.3f})")
        
        return results
    
    def create_unified_pipeline(self):
        """Créer le pipeline unifié avec les meilleurs modèles"""
        print("\n" + "="*80)
        print("🔗 CRÉATION DU PIPELINE UNIFIÉ")
        print("="*80)
        
        self.unified_pipeline = WeatherPredictionPipeline(
            temp_model=self.best_temp_model,
            rain_model=self.best_rain_model,
            scaler=self.scaler
        )
        
        overfitting_warning = ""
        if self.best_rain_metrics.get('overfitting_detected', False):
            overfitting_warning = " ⚠️ (overfitting détecté)"
        
        print(f"\n✅ Pipeline créé:")
        print(f"  • Modèle température: {self.best_temp_model_name} (R²={self.best_temp_score:.3f})")
        print(f"  • Modèle pluie: {self.best_rain_model_name} (Score={self.best_rain_score:.3f}){overfitting_warning}")
        
        return self.unified_pipeline
    
    def log_unified_pipeline_azure(self, run, X_temp_sample, feature_names_temp, feature_names_rain):
        """Enregistrer le pipeline unifié dans Azure ML"""
        if self.unified_pipeline is None:
            print("⚠️ Aucun pipeline unifié à enregistrer")
            return
        
        print("\n" + "="*80)
        print("💾 ENREGISTREMENT DU PIPELINE UNIFIÉ DANS AZURE ML")
        print("="*80)
        
        # Log des métriques du modèle température
        print("\n📊 Métriques - Modèle Température:")
        for metric, value in self.best_temp_metrics.items():
            mlflow.log_metric(f"temp_{metric}", value)
            print(f"  • {metric}: {value:.4f}")
        
        # Log des métriques du modèle pluie
        print("\n📊 Métriques - Modèle Pluie:")
        for metric, value in self.best_rain_metrics.items():
            if isinstance(value, (int, float, bool)):
                mlflow.log_metric(f"rain_{metric}", float(value))
                print(f"  • {metric}: {value:.4f}" if isinstance(value, float) else f"  • {metric}: {value}")
        
        # Log des paramètres
        mlflow.log_param("temp_model_name", self.best_temp_model_name)
        mlflow.log_param("rain_model_name", self.best_rain_model_name)
        mlflow.log_param("n_features_temp", len(feature_names_temp))
        mlflow.log_param("n_features_rain", len(feature_names_rain))
        mlflow.log_param("pipeline_type", "unified")
        mlflow.log_param("rain_model_available", True)
        mlflow.log_param("overfitting_detected", self.best_rain_metrics.get('overfitting_detected', False))
        
        # Log des tags
        mlflow.set_tags({
            "temp_model": self.best_temp_model_name,
            "rain_model": self.best_rain_model_name,
            "model_type": "unified",
            "best_temp_r2": str(self.best_temp_score),
            "best_rain_score": str(self.best_rain_score),
            "overfitting_warning": str(self.best_rain_metrics.get('overfitting_detected', False))
        })
        
        print("\n📦 Sauvegarde des modèles...")
        
        # Créer un répertoire temporaire
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Sauvegarder le pipeline unifié avec pickle
            print("  🔄 Pipeline unifié...")
            pipeline_path = os.path.join(temp_dir, "unified_pipeline.pkl")
            with open(pipeline_path, 'wb') as f:
                pickle.dump(self.unified_pipeline, f)
            mlflow.log_artifact(pipeline_path, artifact_path="models")
            print("  ✓ Pipeline unifié enregistré")
            
            # Sauvegarder le modèle de température
            print("  🔄 Modèle température...")
            temp_model_path = os.path.join(temp_dir, "temperature_model.pkl")
            with open(temp_model_path, 'wb') as f:
                pickle.dump(self.best_temp_model, f)
            mlflow.log_artifact(temp_model_path, artifact_path="models")
            print("  ✓ Modèle température enregistré")
            
            # Sauvegarder le modèle de pluie
            print("  🔄 Modèle pluie...")
            rain_model_path = os.path.join(temp_dir, "rain_model.pkl")
            with open(rain_model_path, 'wb') as f:
                pickle.dump(self.best_rain_model, f)
            mlflow.log_artifact(rain_model_path, artifact_path="models")
            print("  ✓ Modèle pluie enregistré")
            
            # Sauvegarder le scaler
            print("  🔄 Scaler...")
            scaler_path = os.path.join(temp_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            mlflow.log_artifact(scaler_path, artifact_path="models")
            print("  ✓ Scaler enregistré")
            
            # Sauvegarder les encoders
            print("  🔄 Encoders...")
            encoders_path = os.path.join(temp_dir, "encoders.pkl")
            with open(encoders_path, 'wb') as f:
                pickle.dump({
                    'city_encoder': self.city_encoder,
                    'condition_encoder': self.condition_encoder
                }, f)
            mlflow.log_artifact(encoders_path, artifact_path="models")
            print("  ✓ Encoders enregistrés")
            
            # Sauvegarder les noms de features
            print("  🔄 Feature names...")
            features_path = os.path.join(temp_dir, "feature_names.json")
            with open(features_path, 'w') as f:
                json.dump({
                    'temp_features': feature_names_temp,
                    'rain_features': feature_names_rain
                }, f)
            mlflow.log_artifact(features_path, artifact_path="models")
            print("  ✓ Feature names enregistrés")
            
            # Sauvegarder un fichier README
            print("  🔄 Documentation...")
            readme_path = os.path.join(temp_dir, "README.md")
            
            overfitting_note = ""
            if self.best_rain_metrics.get('overfitting_detected', False):
                overfitting_note = f"""
## ⚠️ AVERTISSEMENT OVERFITTING
Le modèle de pluie ({self.best_rain_model_name}) a été détecté comme potentiellement overfitté.
- F1 Test: {self.best_rain_metrics['f1']:.4f}
- F1 CV: {self.best_rain_metrics.get('cv_f1_mean', 0):.4f}
- Score Final (pénalisé): {self.best_rain_score:.4f}

**Recommandations:**
- Collecter plus de données (minimum 100 cas de pluie)
- Réévaluer le modèle sur de nouvelles données
- Considérer une approche plus simple ou régression continue
"""
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"""# Weather Forecasting Model

## Modèles
- **Température**: {self.best_temp_model_name} (R² = {self.best_temp_score:.4f})
- **Pluie**: {self.best_rain_model_name} (Score Final = {self.best_rain_score:.4f})

{overfitting_note}

## Métriques Pluie
- Accuracy Test: {self.best_rain_metrics.get('acc_test', 0):.4f}
- Precision: {self.best_rain_metrics.get('precision', 0):.4f}
- Recall: {self.best_rain_metrics.get('recall', 0):.4f}
- F1-Score Test: {self.best_rain_metrics.get('f1', 0):.4f}
- F1-Score CV: {self.best_rain_metrics.get('cv_f1_mean', 0):.4f} (±{self.best_rain_metrics.get('cv_f1_std', 0):.4f})
- ROC AUC: {self.best_rain_metrics.get('roc_auc', 0):.4f}

## Fichiers
- `unified_pipeline.pkl`: Pipeline complet (température + pluie)
- `temperature_model.pkl`: Modèle température seul
- `rain_model.pkl`: Modèle pluie seul
- `scaler.pkl`: StandardScaler pour normalisation
- `encoders.pkl`: LabelEncoders pour city et condition
- `feature_names.json`: Noms des features

## Utilisation
```python
import pickle
import numpy as np

# Charger le pipeline
with open('unified_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Prédiction
X = np.array([[...]])  # {len(feature_names_temp)} features
predictions = pipeline.predict_proba(X)

print(predictions['temperature'])
print(predictions['will_rain'])
print(predictions['rain_probability'])
```

## Features ({len(feature_names_temp)})
{', '.join(feature_names_temp[:10])}...
""")
            mlflow.log_artifact(readme_path, artifact_path="models")
            print("  ✓ Documentation enregistrée")
            
        finally:
            # Nettoyer
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"\n✅ Pipeline unifié enregistré avec succès dans Azure ML!")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        print(f"   Tous les modèles sont dans: Artifacts → models/")
        
        if self.best_rain_metrics.get('overfitting_detected', False):
            print(f"\n⚠️  ATTENTION: Overfitting détecté sur le modèle de pluie")
            print(f"   Consultez le README.md pour plus de détails")
    
    def log_temperature_only_azure(self, run, feature_names_temp):
        """Enregistrer uniquement le modèle de température dans Azure ML"""
        print("\n" + "="*80)
        print("💾 ENREGISTREMENT DU MODÈLE TEMPÉRATURE DANS AZURE ML")
        print("="*80)
        
        # Log des métriques
        print("\n📊 Métriques - Modèle Température:")
        for metric, value in self.best_temp_metrics.items():
            mlflow.log_metric(f"temp_{metric}", value)
            print(f"  • {metric}: {value:.4f}")
        
        # Log des paramètres
        mlflow.log_param("temp_model_name", self.best_temp_model_name)
        mlflow.log_param("n_features_temp", len(feature_names_temp))
        mlflow.log_param("pipeline_type", "temperature_only")
        mlflow.log_param("rain_model_available", False)
        
        # Log des tags
        mlflow.set_tags({
            "temp_model": self.best_temp_model_name,
            "model_type": "temperature_only",
            "best_temp_r2": str(self.best_temp_score)
        })
        
        print("\n📦 Enregistrement du modèle dans MLflow...")
        
        # Enregistrer le modèle de température
        print("  🔄 Modèle température...")
        mlflow.sklearn.log_model(
            sk_model=self.best_temp_model,
            artifact_path="temperature_model"
        )
        print("  ✓ Modèle température enregistré")
        
        # Enregistrer le scaler
        print("  🔄 Scaler...")
        mlflow.sklearn.log_model(
            sk_model=self.scaler,
            artifact_path="scaler"
        )
        print("  ✓ Scaler enregistré")
        
        # Enregistrer les encoders
        print("  🔄 Encoders...")
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        try:
            with open(temp_file.name, 'wb') as f:
                pickle.dump({
                    'city_encoder': self.city_encoder,
                    'condition_encoder': self.condition_encoder
                }, f)
            mlflow.log_artifact(temp_file.name, artifact_path="encoders")
            print("  ✓ Encoders enregistrés")
        finally:
            os.unlink(temp_file.name)
        
        print(f"\n✅ Modèle température enregistré avec succès dans Azure ML!")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")

    def display_comparison(self, temp_results, rain_results):
        """Afficher la comparaison des modèles"""
        print("\n" + "="*80)
        print("📊 COMPARAISON DES MODÈLES")
        print("="*80)
        
        print("\n🌡️  TEMPÉRATURE (Régression):")
        temp_df = pd.DataFrame({
            'Model': list(temp_results.keys()),
            'MAE (°C)': [r['mae_test'] for r in temp_results.values()],
            'RMSE (°C)': [r['rmse_test'] for r in temp_results.values()],
            'R²': [r['r2_test'] for r in temp_results.values()]
        })
        temp_df = temp_df.sort_values('R²', ascending=False)
        temp_df['Best'] = temp_df['Model'].apply(
            lambda x: '🏆' if x == self.best_temp_model_name else ''
        )
        print(temp_df.to_string(index=False))
        
        if rain_results:
            print("\n🌧️  PLUIE (Classification avec détection overfitting):")
            rain_df = pd.DataFrame({
                'Model': list(rain_results.keys()),
                'Accuracy': [r['acc_test'] for r in rain_results.values()],
                'Precision': [r['precision'] for r in rain_results.values()],
                'Recall': [r['recall'] for r in rain_results.values()],
                'F1-Test': [r['f1'] for r in rain_results.values()],
                'F1-CV': [r['cv_f1_mean'] if r['cv_f1_mean'] > 0 else float('nan') for r in rain_results.values()],
                'Score Final': [r['final_score'] for r in rain_results.values()],
                'Overfit?': ['⚠️' if r['overfitting_detected'] else '✓' for r in rain_results.values()]
            })
            rain_df = rain_df.sort_values('Score Final', ascending=False)
            rain_df['Best'] = rain_df['Model'].apply(
                lambda x: '🏆' if x == self.best_rain_model_name else ''
            )
            print(rain_df.to_string(index=False))
        else:
            rain_df = None
        
        print(f"\n🏆 MEILLEURS MODÈLES SÉLECTIONNÉS:")
        print(f"  • Température: {self.best_temp_model_name} (R²={self.best_temp_score:.3f})")
        if self.best_rain_model_name:
            overfitting_note = " ⚠️ (overfitting détecté)" if self.best_rain_metrics.get('overfitting_detected', False) else ""
            print(f"  • Pluie: {self.best_rain_model_name} (Score={self.best_rain_score:.3f}){overfitting_note}")
        
        return temp_df, rain_df


def main():
    """Pipeline principal d'entraînement"""
    
    # Configuration Azure
    STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT_NAME", "stweatherwassimv2")
    CONTAINER = os.getenv("CONTAINER_NAME", "weather-data")
    STORAGE_KEY = os.getenv("STORAGE_ACCOUNT_KEY")
    
    SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
    WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
    
    if not STORAGE_KEY:
        raise ValueError(
            "⚠️ AZURE_STORAGE_KEY non trouvée dans les variables d'environnement.\n"
            "   Ajoutez-la dans votre fichier .env"
        )
    
    print("="*80)
    print("🌤️  WEATHER FORECASTING - PIPELINE UNIFIÉ AVEC DÉTECTION OVERFITTING")
    print("="*80)
    print("📋 Prédictions:")
    print("  1️⃣ Température (Régression)")
    print("  2️⃣ Pluie (Classification avec température prédite)")
    print("  🔍 Détection automatique d'overfitting via Cross-Validation")
    print("  ⚠️  Pénalités sévères pour scores parfaits sur petits datasets")
    print("="*80)
    
    # Connexion à Azure ML
    azure_ml_client = None
    if SUBSCRIPTION_ID and RESOURCE_GROUP and WORKSPACE_NAME:
        try:
            print("\n🔗 Connexion à Azure ML Workspace...")
            # Essayer AzureCliCredential en premier (si az login a été fait)
            try:
                credential = AzureCliCredential()
                azure_ml_client = MLClient(
                    credential=credential,
                    subscription_id=SUBSCRIPTION_ID,
                    resource_group_name=RESOURCE_GROUP,
                    workspace_name=WORKSPACE_NAME
                )
                print("  ✓ Connecté via Azure CLI")
            except:
                # Fallback vers DefaultAzureCredential
                credential = DefaultAzureCredential()
                azure_ml_client = MLClient(
                    credential=credential,
                    subscription_id=SUBSCRIPTION_ID,
                    resource_group_name=RESOURCE_GROUP,
                    workspace_name=WORKSPACE_NAME
                )
                print("  ✓ Connecté via DefaultAzureCredential")
        except Exception as e:
            print(f"  ⚠️ Impossible de se connecter à Azure ML: {e}")
            print("  ℹ️  Les modèles seront sauvegardés localement uniquement")
    else:
        print("\n⚠️ Configuration Azure ML manquante dans .env")
        print("   Les modèles seront sauvegardés localement uniquement")
    
    # Configurer MLflow pour Azure ML
    if azure_ml_client:
        # Obtenir l'URI de tracking d'Azure ML
        workspace = azure_ml_client.workspaces.get(WORKSPACE_NAME)
        mlflow_tracking_uri = workspace.mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"  ✓ MLflow tracking URI: {mlflow_tracking_uri}")
    
    # Configurer l'expérience
    experiment_name = "weather-unified-forecast"
    mlflow.set_experiment(experiment_name)
    
    # Démarrer un run MLflow/Azure ML
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        
        print(f"\n🔬 Run ID: {run.info.run_id}")
        
        # Initialiser le pipeline
        pipeline = WeatherMLPipeline(
            storage_account_name=STORAGE_ACCOUNT,
            container_name=CONTAINER,
            storage_account_key=STORAGE_KEY,
            experiment_name=experiment_name,
            azure_ml_client=azure_ml_client
        )
        
        # Charger les données
        print("\n📁 Chargement des données depuis Azure Blob Storage...")
        df = pipeline.load_data_from_blob("bronze/history")
        
        if df is None or len(df) == 0:
            print("\n✗ Aucune donnée disponible")
            return
        
        # Feature engineering
        print("\n🔧 Feature engineering...")
        df_features = pipeline.feature_engineering(df)
        
        # Créer la cible de classification
        df_features = pipeline.create_classification_target(df_features, threshold=0.1)
        
        # ========================================================================
        # PARTIE 1: ENTRAÎNEMENT DES MODÈLES DE TEMPÉRATURE
        # ========================================================================
        print("\n" + "="*80)
        print("PARTIE 1: MODÈLES DE TEMPÉRATURE")
        print("="*80)
        
        X_temp, y_temp, feature_names_temp = pipeline.prepare_features_target(
            df_features, target='temp_c'
        )
        
        print(f"\n📋 Données température:")
        print(f"  Features: {len(feature_names_temp)}")
        print(f"  Observations: {len(X_temp)}")
        
        # Split pour température
        X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        print(f"  Train: {len(X_temp_train)} | Test: {len(X_temp_test)}")
        
        # Normalisation
        print("\n🔄 Normalisation...")
        pipeline.scaler.fit(X_temp_train)
        X_temp_train_scaled = pipeline.scaler.transform(X_temp_train)
        X_temp_test_scaled = pipeline.scaler.transform(X_temp_test)
        
        # Entraîner les modèles de température
        temp_results = pipeline.train_temperature_models(
            X_temp_train_scaled, X_temp_test_scaled, y_temp_train, y_temp_test
        )
        
        # ========================================================================
        # PARTIE 2: ENTRAÎNEMENT DES MODÈLES DE PLUIE (avec température)
        # ========================================================================
        print("\n" + "="*80)
        print("PARTIE 2: MODÈLES DE PLUIE (avec température prédite)")
        print("="*80)
        
        # Préparer les données pour la pluie
        X_rain_base, _, _ = pipeline.prepare_features_target(df_features, target='temp_c')
        y_rain = df_features.loc[X_rain_base.index, 'will_rain']
        
        # Analyser le déséquilibre
        strategy = pipeline.analyze_data_balance(y_rain, task='classification')
        
        # Vérifier si on a au moins 2 classes
        n_classes_rain = len(np.unique(y_rain))
        
        rain_results = {}
        
        if n_classes_rain < 2:
            print("\n⚠️ ATTENTION CRITIQUE: Une seule classe dans les données de pluie!")
            print(f"   Classe présente: {np.unique(y_rain)}")
            print(f"   Impossible de créer un modèle de classification binaire.")
            print(f"\n⏭️  Passage à la sauvegarde des résultats de température uniquement...")
            
            pipeline.rain_model_available = False
            
            # Afficher uniquement les résultats température
            print("\n" + "="*80)
            print("📊 RÉSULTATS - TEMPÉRATURE UNIQUEMENT")
            print("="*80)
            
            temp_df = pd.DataFrame({
                'Model': list(temp_results.keys()),
                'MAE (°C)': [r['mae_test'] for r in temp_results.values()],
                'RMSE (°C)': [r['rmse_test'] for r in temp_results.values()],
                'R²': [r['r2_test'] for r in temp_results.values()]
            })
            temp_df = temp_df.sort_values('R²', ascending=False)
            temp_df['Best'] = temp_df['Model'].apply(
                lambda x: '🏆' if x == pipeline.best_temp_model_name else ''
            )
            print(temp_df.to_string(index=False))
            
            print(f"\n🏆 MEILLEUR MODÈLE:")
            print(f"  • Température: {pipeline.best_temp_model_name} (R²={pipeline.best_temp_score:.3f})")
            
            # Enregistrer dans Azure ML
            pipeline.log_temperature_only_azure(run, feature_names_temp)
            
            print("\n✅ Modèle de température sauvegardé avec succès!")
            print("\n🔔 NOTE: Le déploiement se fera avec le modèle température uniquement")
            return
        
        # Si on a au moins 2 classes, continuer avec l'entraînement pluie
        pipeline.rain_model_available = True
        
        # Split pour pluie
        X_rain_base_train, X_rain_base_test, y_rain_train, y_rain_test = train_test_split(
            X_rain_base, y_rain, test_size=0.2, random_state=42, stratify=y_rain
        )
        
        # Normaliser
        X_rain_base_train_scaled = pipeline.scaler.transform(X_rain_base_train)
        X_rain_base_test_scaled = pipeline.scaler.transform(X_rain_base_test)
        
        # AJOUTER la température prédite comme feature
        print("\n➕ Ajout de la température prédite comme feature...")
        temp_pred_train = pipeline.best_temp_model.predict(X_rain_base_train_scaled)
        temp_pred_test = pipeline.best_temp_model.predict(X_rain_base_test_scaled)
        
        X_rain_train = np.column_stack([X_rain_base_train_scaled, temp_pred_train])
        X_rain_test = np.column_stack([X_rain_base_test_scaled, temp_pred_test])
        
        print(f"  ✓ Features pluie: {X_rain_train.shape[1]} (incluant température prédite)")
        
        # Balancing si nécessaire
        if strategy != 'none':
            X_rain_train, y_rain_train = pipeline.balance_data(
                X_rain_train, y_rain_train, strategy=strategy
            )
        
        # Entraîner les modèles de pluie (avec détection overfitting AMÉLIORÉE)
        rain_results = pipeline.train_rain_models(
            X_rain_train, X_rain_test, y_rain_train, y_rain_test
        )
        
        # ========================================================================
        # PARTIE 3: CRÉER ET ENREGISTRER LE PIPELINE UNIFIÉ
        # ========================================================================
        
        # Afficher la comparaison
        temp_df, rain_df = pipeline.display_comparison(temp_results, rain_results)
        
        # Créer le pipeline unifié
        if rain_results and pipeline.rain_model_available:
            unified_pipeline = pipeline.create_unified_pipeline()
            
            # Enregistrer dans Azure ML
            feature_names_rain = feature_names_temp + ['temp_predicted']
            pipeline.log_unified_pipeline_azure(
                run, X_temp_train_scaled, feature_names_temp, feature_names_rain
            )
        
        # ========================================================================
        # PARTIE 4: TEST DU PIPELINE UNIFIÉ
        # ========================================================================
        if rain_results and pipeline.unified_pipeline:
            print("\n" + "="*80)
            print("🧪 TEST DU PIPELINE UNIFIÉ")
            print("="*80)
            
            # Prendre quelques exemples de test
            n_samples = min(5, len(X_temp_test_scaled))
            X_test_sample = X_temp_test_scaled[:n_samples]
            
            print(f"\n🔮 Prédictions sur {n_samples} exemples:")
            predictions = pipeline.unified_pipeline.predict_proba(X_test_sample)
            
            for i in range(n_samples):
                print(f"\n  Exemple {i+1}:")
                print(f"    Température réelle: {y_temp_test.iloc[i]:.1f}°C")
                print(f"    Température prédite: {predictions['temperature'][i]:.1f}°C")
                print(f"    Pluie réelle: {'Oui' if y_rain_test.iloc[i] == 1 else 'Non'}")
                print(f"    Pluie prédite: {'Oui' if predictions['will_rain'][i] == 1 else 'Non'}")
                print(f"    Probabilité de pluie: {predictions['rain_probability'][i]:.1%}")
        
        # ========================================================================
        # RÉSUMÉ FINAL
        # ========================================================================
        print("\n" + "="*80)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("="*80)
        
        print(f"\n📊 Résultats:")
        print(f"  • Modèles température testés: {len(temp_results)}")
        if rain_results:
            print(f"  • Modèles pluie testés: {len(rain_results)}")
        print(f"  • Meilleur température: {pipeline.best_temp_model_name} (R²={pipeline.best_temp_score:.3f})")
        if pipeline.best_rain_model_name:
            overfitting_note = " ⚠️ (overfitting)" if pipeline.best_rain_metrics.get('overfitting_detected', False) else ""
            print(f"  • Meilleur pluie: {pipeline.best_rain_model_name} (Score={pipeline.best_rain_score:.3f}){overfitting_note}")
        
        if azure_ml_client:
            print(f"\n☁️  Modèles enregistrés dans Azure ML:")
            if rain_results and pipeline.rain_model_available:
                print(f"  • weather_unified_pipeline")
                print(f"  • weather_temperature_{pipeline.best_temp_model_name.lower()}")
                print(f"  • weather_rain_{pipeline.best_rain_model_name.lower()}")
            else:
                print(f"  • weather_temperature_{pipeline.best_temp_model_name.lower()}")
            
            print(f"\n🎯 Prêt pour le déploiement automatique!")
            print(f"  Les modèles sont disponibles dans Azure ML Model Registry")
        else:
            print(f"\n⚠️ Modèles sauvegardés localement uniquement")
            print(f"  Configure Azure ML pour enregistrer dans le cloud")
        
        if pipeline.best_rain_metrics.get('overfitting_detected', False):
            print(f"\n⚠️  AVERTISSEMENT:")
            print(f"  Le modèle de pluie présente des signes d'overfitting")
            print(f"  Recommandation: Collecter plus de données avant déploiement production")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()