import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import toml

st.set_page_config(
    page_title="Prévisions Météo - Azure ML",
    page_icon="🌤️",
    layout="wide"
)

# Styles CSS améliorés
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .current-weather {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour récupérer les données météo actuelles
@st.cache_data(ttl=600)  # Cache pendant 10 minutes
def get_current_weather(city, weather_api_key):
    """Récupère les données météo actuelles via WeatherAPI"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}&aqi=no"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données météo: {str(e)}")
        return None

# Fonction pour préparer les features depuis les données météo
def prepare_features_from_weather(weather_data, hour_offset=0):
    """Prépare les features pour le modèle ML à partir des données météo"""
    current = weather_data['current']
    location = weather_data['location']
    
    # Calculer l'heure cible
    local_time = datetime.strptime(location['localtime'], "%Y-%m-%d %H:%M")
    target_time = local_time + timedelta(hours=hour_offset)
    
    hour = target_time.hour
    day_of_week = target_time.weekday()
    month = target_time.month
    
    # Features temporelles
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    is_weekend = 1 if day_of_week >= 5 else 0
    is_day = 1 if 6 <= hour <= 20 else 0
    
    # Encoder city
    city_name = location['name']
    city_map = {
        "Casablanca": 0, "Rabat": 1, "Marrakech": 2, 
        "Tangier": 3, "Fes": 4, "Agadir": 5,
        "Tanger": 3, "Fès": 4  # Variantes
    }
    city_encoded = city_map.get(city_name, 0)
    
    # Extraire les données météo
    temp_c = current['temp_c']
    humidity = current['humidity']
    wind_kph = current['wind_kph']
    wind_degree = current['wind_degree']
    pressure_mb = current['pressure_mb']
    cloud_cover = current['cloud']
    uv_index = current['uv']
    vis_km = current['vis_km']
    
    # Features d'interaction
    temp_humidity_int = temp_c * humidity / 100
    wind_temp_int = wind_kph * temp_c
    
    # Features lag (utilisées les valeurs actuelles comme estimation)
    temp_lag_1 = temp_c
    temp_lag_2 = temp_c - 0.5
    temp_lag_3 = temp_c - 1.0
    
    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "city_encoded": city_encoded,
        "condition_encoded": 1,
        "is_day": is_day,
        "wind_kph": float(wind_kph),
        "wind_degree": wind_degree,
        "pressure_mb": pressure_mb,
        "humidity": humidity,
        "cloud_cover": cloud_cover,
        "uv_index": uv_index,
        "vis_km": float(vis_km),
        "temp_humidity_interaction": temp_humidity_int,
        "wind_temp_interaction": wind_temp_int,
        "temp_lag_1": temp_lag_1,
        "temp_lag_2": temp_lag_2,
        "temp_lag_3": temp_lag_3,
        "precip_lag_1": 0.0,
        "precip_lag_2": 0.0,
        "precip_lag_3": 0.0
    }, target_time

# Fonction pour faire une prédiction (VERSION CORRIGÉE POUR TON API)
def predict_temperature(features, endpoint_url, api_key):
    """Appelle l'API Azure ML pour prédire la température"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {"data": [features]}
        
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        # Récupérer la réponse
        result = response.json()
        
        # ⚠️ TON API RETOURNE UNE CHAÎNE JSON DANS UN JSON
        # Format: "{\"success\": true, \"predictions\": [{\"temperature_celsius\": 20.5, ...}]}"
        
        # Si result est une chaîne, la parser
        if isinstance(result, str):
            result = json.loads(result)
        
        # Vérifier le succès
        if not result.get('success', False):
            st.error("❌ L'API a retourné success=false")
            return None
        
        # Extraire la température depuis predictions
        predictions = result.get('predictions', [])
        
        if not predictions or len(predictions) == 0:
            st.error("❌ Aucune prédiction dans la réponse")
            return None
        
        # Récupérer la première prédiction
        first_prediction = predictions[0]
        
        # Extraire la température
        temperature = first_prediction.get('temperature_celsius')
        
        if temperature is None:
            st.error("❌ Température non trouvée dans la réponse")
            with st.expander("🔍 Contenu de la prédiction"):
                st.json(first_prediction)
            return None
        
        # ⚠️ PROBLÈME DÉTECTÉ: Ta température est négative (-144°C) !
        # Cela indique probablement un problème avec le modèle ou les features
        if temperature < -50 or temperature > 60:
            st.warning(f"⚠️ Température suspecte: {temperature:.1f}°C")
            st.info("💡 Le modèle pourrait nécessiter un ré-entraînement avec de meilleures données")
        
        return float(temperature)
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout: L'API met trop de temps à répondre")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Erreur de connexion: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"📄 Erreur JSON: {str(e)}")
        with st.expander("🔍 Réponse brute"):
            st.code(response.text[:500])
        return None
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        with st.expander("🔍 Détails"):
            import traceback
            st.code(traceback.format_exc())
        return None

st.markdown('<h1 class="main-header">🌤️ Prévisions Météorologiques en Temps Réel</h1>', unsafe_allow_html=True)
st.markdown("**Données actuelles + Prédictions IA via Azure Machine Learning**")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Charger les credentials
    secrets_path = Path(__file__).parent / "secrets.toml"
    
    if secrets_path.exists():
        try:
            secrets = toml.load(secrets_path)
            endpoint_url = secrets.get("ENDPOINT_URL", "")
            api_key = secrets.get("API_KEY", "")
            weather_api_key = secrets.get("WEATHER_API_KEY", "")
            
            if endpoint_url and api_key:
                st.success("✅ Azure ML connecté")
            else:
                st.error("❌ Configuration Azure ML incomplète")
                st.stop()
            
            if not weather_api_key:
                st.warning("⚠️ Clé API météo manquante")
                st.info("Inscrivez-vous sur weatherapi.com pour obtenir une clé gratuite")
                
        except Exception as e:
            st.error(f"❌ Erreur configuration: {str(e)}")
            st.stop()
    else:
        st.error("🔑 Fichier secrets.toml introuvable")
        st.stop()
    
    st.markdown("---")
    
    # Sélection de la ville
    st.subheader("📍 Localisation")
    city = st.selectbox(
        "Ville marocaine",
        ["Casablanca", "Rabat", "Marrakech", "Tangier", "Fes", "Agadir"],
        help="Sélectionnez votre ville"
    )
    
    # Options de prédiction
    st.subheader("🔮 Prédictions")
    prediction_hours = st.multiselect(
        "Heures à prédire",
        [1, 2, 3, 6, 12, 24],
        default=[1, 3, 6],
        help="Sélectionnez les horizons de prédiction"
    )
    
    # Bouton de rafraîchissement
    if st.button("🔄 Actualiser", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 À propos")
    st.info("""
    **Projet:** Météo + IA  
    **Technologie:** Azure ML  
    **Données:** Temps réel  
    **École:** UIR 2025
    """)

# Interface principale avec tabs
tabs = st.tabs(["🌍 Tableau de Bord", "📊 Analyse", "ℹ️ Documentation"])

with tabs[0]:
    # ========================================================================
    # SECTION 1: MÉTÉO ACTUELLE
    # ========================================================================
    st.header(f"🌍 Météo Actuelle - {city}")
    
    # Récupérer les données météo
    weather_data = None
    if weather_api_key:
        with st.spinner("🔄 Récupération des données météo..."):
            weather_data = get_current_weather(city, weather_api_key)
        
        if weather_data:
            current = weather_data['current']
            location = weather_data['location']
            
            # Affichage principal avec deux colonnes
            col_main1, col_main2 = st.columns([2, 1])
            
            with col_main1:
                # Carte principale de météo actuelle
                st.markdown(f"""
                <div class="current-weather">
                    <h2>🌡️ {current['temp_c']:.1f}°C</h2>
                    <h3>{current['condition']['text']}</h3>
                    <p style="font-size: 1.2em;">📍 {location['name']}, {location['country']}</p>
                    <p style="font-size: 1.1em;">🕐 {location['localtime']}</p>
                    <p style="font-size: 1em; margin-top: 1rem;">Ressenti: {current['feelslike_c']:.1f}°C</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_main2:
                # Jauge de température
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current['temp_c'],
                    title={'text': "Température", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [-10, 50], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-10, 0], 'color': "lightblue"},
                            {'range': [0, 15], 'color': "lightyellow"},
                            {'range': [15, 25], 'color': "lightgreen"},
                            {'range': [25, 35], 'color': "orange"},
                            {'range': [35, 50], 'color': "red"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Métriques détaillées
            st.subheader("📊 Conditions Détaillées")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💧 Humidité", f"{current['humidity']}%")
                st.metric("☁️ Nuages", f"{current['cloud']}%")
            
            with col2:
                st.metric("💨 Vent", f"{current['wind_kph']:.0f} km/h")
                st.metric("🧭 Direction", f"{current['wind_dir']}")
            
            with col3:
                st.metric("🧭 Pression", f"{current['pressure_mb']:.0f} mb")
                st.metric("👁️ Visibilité", f"{current['vis_km']:.0f} km")
            
            with col4:
                st.metric("🌅 Index UV", f"{current['uv']}")
                st.metric("🌧️ Précip.", f"{current['precip_mm']:.1f} mm")
            
            # ========================================================================
            # SECTION 2: PRÉDICTIONS IA
            # ========================================================================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.header("🔮 Prédictions Intelligence Artificielle")
            st.info(f"🤖 Prédictions générées par Azure ML | Base: données actuelles à {location['localtime']}")
            
            if prediction_hours:
                # Faire les prédictions
                predictions = []
                
                with st.spinner("🔄 Calcul des prédictions IA en cours..."):
                    for hours in sorted(prediction_hours):
                        features, target_time = prepare_features_from_weather(weather_data, hour_offset=hours)
                        predicted_temp = predict_temperature(features, endpoint_url, api_key)
                        
                        if predicted_temp is not None:
                            predictions.append({
                                'hours': hours,
                                'target_time': target_time,
                                'temperature': predicted_temp,
                                'current_temp': current['temp_c']
                            })
                
                if predictions:
                    # Cartes de prédictions
                    st.subheader("📅 Prévisions par Heure")
                    
                    cols = st.columns(len(predictions))
                    
                    for i, pred in enumerate(predictions):
                        with cols[i]:
                            delta = pred['temperature'] - pred['current_temp']
                            delta_text = f"{delta:+.1f}°C"
                            delta_emoji = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                            
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>Dans {pred['hours']}h</h3>
                                <h1 style="margin: 0.5rem 0;">{pred['temperature']:.1f}°C</h1>
                                <p style="font-size: 1.1em;">🕐 {pred['target_time'].strftime('%H:%M')}</p>
                                <p style="font-size: 1.3em; margin-top: 0.5rem;">{delta_emoji} {delta_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Graphique d'évolution
                    st.markdown("---")
                    st.subheader("📈 Évolution de la Température")
                    
                    df_pred = pd.DataFrame(predictions)
                    
                    # Ajouter le point actuel
                    current_point = {
                        'hours': 0,
                        'target_time': datetime.strptime(location['localtime'], "%Y-%m-%d %H:%M"),
                        'temperature': current['temp_c'],
                        'current_temp': current['temp_c']
                    }
                    df_full = pd.concat([pd.DataFrame([current_point]), df_pred], ignore_index=True)
                    
                    fig = go.Figure()
                    
                    # Ligne de température actuelle (référence)
                    fig.add_trace(go.Scatter(
                        x=df_full['hours'],
                        y=[current['temp_c']] * len(df_full),
                        mode='lines',
                        name='Température actuelle',
                        line=dict(color='gray', width=2, dash='dash'),
                        opacity=0.5
                    ))
                    
                    # Ligne de prédiction
                    fig.add_trace(go.Scatter(
                        x=df_full['hours'],
                        y=df_full['temperature'],
                        mode='lines+markers',
                        name='Température prédite',
                        line=dict(color='#f5576c', width=4),
                        marker=dict(size=12, symbol='circle', line=dict(color='white', width=2)),
                        fill='tonexty',
                        fillcolor='rgba(245, 87, 108, 0.1)'
                    ))
                    
                    fig.update_layout(
                        title={
                            'text': f"Prévisions de température pour {city}",
                            'font': {'size': 20}
                        },
                        xaxis_title="Heures à partir de maintenant",
                        yaxis_title="Température (°C)",
                        hovermode='x unified',
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau des prédictions
                    col_table, col_reco = st.columns([2, 1])
                    
                    with col_table:
                        st.subheader("📋 Détails des Prédictions")
                        
                        df_display = df_pred.copy()
                        df_display['Heure'] = df_display['target_time'].dt.strftime('%H:%M')
                        df_display['Température'] = df_display['temperature'].apply(lambda x: f"{x:.1f}°C")
                        df_display['Évolution'] = (df_display['temperature'] - df_display['current_temp']).apply(
                            lambda x: f"{'🔥' if x > 2 else '❄️' if x < -2 else '➡️'} {x:+.1f}°C"
                        )
                        
                        st.dataframe(
                            df_display[['hours', 'Heure', 'Température', 'Évolution']].rename(columns={'hours': 'Dans (h)'}),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col_reco:
                        st.subheader("💡 Recommandations")
                        
                        max_temp = max([p['temperature'] for p in predictions])
                        min_temp = min([p['temperature'] for p in predictions])
                        temp_range = max_temp - min_temp
                        
                        if temp_range > 5:
                            st.warning(f"⚠️ Variation importante:\n{temp_range:.1f}°C")
                            st.info("🧥 Vêtements adaptables recommandés")
                        else:
                            st.success(f"✅ Température stable:\n±{temp_range:.1f}°C")
                        
                        if max_temp > 30:
                            st.error("🌡️ Forte chaleur\n💧 Hydratez-vous!")
                        elif max_temp > 25:
                            st.warning("☀️ Temps chaud\n😎 Protection solaire")
                        elif min_temp < 10:
                            st.info("❄️ Temps frais\n🧥 Couvrez-vous bien")
                        else:
                            st.success("😊 Température agréable")
                        
                        # Recommandation pour la pluie
                        if current['precip_mm'] > 0:
                            st.warning("☔ Pluie détectée\nPrévoyez un parapluie")
                        
                        # Recommandation UV
                        if current['uv'] > 6:
                            st.warning(f"🌅 UV élevé ({current['uv']})\nProtection recommandée")
                
                else:
                    st.error("❌ Impossible de générer les prédictions")
            else:
                st.warning("⚠️ Sélectionnez des heures de prédiction dans la barre latérale")
        
    else:
        st.warning("⚠️ Configurez votre clé API météo dans secrets.toml")
        st.code("""
# Ajoutez dans secrets.toml:
WEATHER_API_KEY = "votre_clé_weatherapi"

# Obtenez une clé gratuite sur:
# https://www.weatherapi.com/signup.aspx
        """)

with tabs[1]:
    st.header("📊 Analyse Comparative")
    
    if weather_data and 'predictions' in locals() and predictions:
        # Comparaison actuel vs prédictions
        st.subheader("📉 Écarts par rapport à maintenant")
        
        df_analysis = pd.DataFrame(predictions)
        df_analysis['delta'] = df_analysis['temperature'] - df_analysis['current_temp']
        
        fig = go.Figure()
        
        colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in df_analysis['delta']]
        
        fig.add_trace(go.Bar(
            x=df_analysis['hours'],
            y=df_analysis['delta'],
            marker_color=colors,
            text=df_analysis['delta'].apply(lambda x: f"{x:+.1f}°C"),
            textposition='outside',
            hovertemplate='Dans %{x}h<br>Écart: %{y:.1f}°C<extra></extra>'
        ))
        
        fig.update_layout(
            title="Écart de température par rapport à maintenant",
            xaxis_title="Heures",
            yaxis_title="Écart (°C)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques
        st.subheader("📈 Statistiques des Prédictions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        max_temp = df_analysis['temperature'].max()
        min_temp = df_analysis['temperature'].min()
        avg_temp = df_analysis['temperature'].mean()
        temp_range = max_temp - min_temp
        
        with col1:
            st.metric("📈 Maximum", f"{max_temp:.1f}°C")
        
        with col2:
            st.metric("📉 Minimum", f"{min_temp:.1f}°C")
        
        with col3:
            st.metric("📊 Moyenne", f"{avg_temp:.1f}°C")
        
        with col4:
            st.metric("📏 Amplitude", f"{temp_range:.1f}°C")
        
        # Comparaison avec la moyenne historique
        st.subheader("🌡️ Comparaison")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            # Gauge comparatif
            fig_comp = go.Figure(go.Indicator(
                mode="number+delta",
                value=avg_temp,
                delta={'reference': current['temp_c'], 'relative': False},
                title={'text': "Température moyenne prédite vs actuelle"},
            ))
            fig_comp.update_layout(height=200)
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col_comp2:
            st.info(f"""
            **Tendance générale:**
            
            {'📈 Réchauffement' if avg_temp > current['temp_c'] else '📉 Refroidissement' if avg_temp < current['temp_c'] else '➡️ Stable'}
            
            Écart moyen: **{avg_temp - current['temp_c']:+.1f}°C**
            """)
    else:
        st.info("📊 Lancez d'abord des prédictions dans le tableau de bord")

with tabs[2]:
    st.header("ℹ️ Documentation du Projet")
    
    st.markdown("""
    ### 🎯 À propos de cette application
    
    Cette application combine :
    - 🌍 **Données météo en temps réel** via WeatherAPI
    - 🤖 **Intelligence artificielle** via Azure Machine Learning
    - 📊 **Visualisations interactives** avec Plotly
    - 🔮 **Prédictions multi-horizons** (1h à 24h)
    
    ### 🔧 Fonctionnalités principales
    
    1. **Météo actuelle** 
       - Température, humidité, vent, pression
       - Conditions atmosphériques détaillées
       - Jauge visuelle de température
    
    2. **Prédictions IA** 
       - Température prédite à différents horizons
       - Évolution graphique
       - Écarts par rapport à maintenant
    
    3. **Recommandations intelligentes**
       - Conseils vestimentaires
       - Alertes chaleur/froid
       - Protection UV
    
    4. **Analyse comparative**
       - Statistiques des prédictions
       - Tendances
       - Amplitudes thermiques
    
    ### 📈 Comment ça marche ?
```
    1. Récupération données météo actuelles (WeatherAPI)
    2. Extraction de 28 features météorologiques
    3. Préparation pour différents horizons temporels
    4. Appel du modèle ML Azure pour chaque horizon
    5. Affichage des résultats + analyses
```
    
    ### 🔑 Configuration requise
    
    **Dans `secrets.toml` :**
```toml
    ENDPOINT_URL = "votre-endpoint-azure-ml"
    API_KEY = "votre-clé-azure"
    WEATHER_API_KEY = "votre-clé-weatherapi"
```
    
    ### 📚 Stack Technologique
    
    | Composant | Technologie |
    |-----------|-------------|
    | Cloud ML | Microsoft Azure ML |
    | Modèle | scikit-learn + MLflow |
    | Interface | Streamlit |
    | Visualisation | Plotly |
    | API Météo | WeatherAPI.com |
    | Langage | Python 3.9+ |
    
    ### 🎓 Modèle d'IA
    
    **Features utilisées (28):**
    - Temporelles: heure, jour, mois, cycliques
    - Localisation: ville encodée
    - Conditions: température, humidité, pression, UV
    - Vent: vitesse, direction
    - Interactions: temp×humidité, vent×temp
    - Lag: valeurs précédentes (1h, 2h, 3h)
    
    **Performance:**
    - Type: Régression (température)
    - Algorithme: Meilleur modèle sélectionné automatiquement
    - Métriques: MAE, RMSE, R²
    
    ### 💡 Conseils d'utilisation
    
    - 🔄 **Actualisez** régulièrement pour des données fraîches
    - 📊 **Sélectionnez** plusieurs horizons pour voir l'évolution
    - 📈 **Consultez** l'onglet Analyse pour les tendances
    - ⚙️ **Changez** de ville dans la barre latérale
    
    ### 👥 Projet réalisé par
    
    **UIR - 5ème année Big Data & AI**  
    **Année universitaire 2024-2025**
    
    ---
    
    *Développé avec ❤️ en utilisant Azure Machine Learning et Streamlit*
    """)

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("**🎓 UIR - 5ème année**")

with col_f2:
    st.markdown("**🤖 Big Data & AI**")

with col_f3:
    st.markdown("**📅 2025**")