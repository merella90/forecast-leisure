import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import warnings
import io
warnings.filterwarnings('ignore')

# Configurazione pagina
st.set_page_config(
    page_title="VOI Alimini - Forecasting ADR BED 2026",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_historical_data(file_2023, file_2024, file_2025):
    """Carica e processa i dati storici dalle tre stagioni"""
    
    try:
        # Carica i tre file
        df_2023 = pd.read_excel(file_2023)
        df_2024 = pd.read_excel(file_2024)
        df_2025 = pd.read_excel(file_2025)
        
        # Verifica che abbiano le colonne necessarie
        required_columns = ['Giorno', 'ADR Bed']
        
        for year, df in [(2023, df_2023), (2024, df_2024), (2025, df_2025)]:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"File {year}: Colonne mancanti: {missing}. Colonne presenti: {list(df.columns)}")
        
        # Aggiungi anno a ciascun dataframe
        df_2023['Anno'] = 2023
        df_2024['Anno'] = 2024
        df_2025['Anno'] = 2025
        
        # Combina i dataframe
        df_all = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
        
        # Parsing della data
        df_all['Data'] = pd.to_datetime(df_all['Giorno'], format='%a %d/%m/%Y', errors='coerce')
        
        # Rimuovi righe senza data valida
        df_all = df_all.dropna(subset=['Data'])
        
        if len(df_all) == 0:
            raise ValueError("Nessuna data valida trovata nei file. Verifica il formato delle date.")
        
        # Estrai informazioni temporali
        df_all['Mese'] = df_all['Data'].dt.month
        df_all['Giorno_Settimana'] = df_all['Data'].dt.dayofweek
        df_all['Settimana_Anno'] = df_all['Data'].dt.isocalendar().week.astype(int)
        df_all['Giorno_Nome'] = df_all['Data'].dt.day_name()
        df_all['Mese_Nome'] = df_all['Data'].dt.month_name()
        
        # Calcola il giorno relativo dall'inizio della stagione
        for anno in [2023, 2024, 2025]:
            df_anno = df_all[df_all['Anno'] == anno]
            if len(df_anno) > 0:
                data_inizio = df_anno['Data'].min()
                giorni_stagione = (df_all.loc[df_all['Anno'] == anno, 'Data'] - data_inizio).dt.days
                df_all.loc[df_all['Anno'] == anno, 'Giorno_Stagione'] = giorni_stagione
        
        # Filtra dati validi (ADR BED > 0)
        df_all = df_all[df_all['ADR Bed'] > 0].copy()
        
        if len(df_all) < 30:
            raise ValueError(f"Dati insufficienti dopo il filtraggio. Solo {len(df_all)} giorni validi trovati (minimo 30 richiesti).")
        
        return df_all
        
    except Exception as e:
        raise Exception(f"Errore nel caricamento dei dati: {str(e)}")

def calculate_seasonality_index(df):
    """Calcola l'indice di stagionalità per settimana"""
    
    # Media complessiva ADR BED
    adr_medio_totale = df['ADR Bed'].mean()
    
    # Media per settimana dell'anno
    seasonality = df.groupby('Settimana_Anno')['ADR Bed'].mean().reset_index()
    seasonality['Indice_Stagionalita'] = seasonality['ADR Bed'] / adr_medio_totale
    
    return seasonality

def build_forecast_models(df):
    """Costruisce diversi modelli di forecasting"""
    
    # Verifica che tutte le colonne necessarie esistano
    required_cols = ['Giorno_Stagione', 'Settimana_Anno', 'Giorno_Settimana', 'Mese', 'ADR Bed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")
    
    # Prepara i dati per il training
    df_train = df[required_cols].copy()
    
    # Rimuovi righe con valori mancanti
    df_train = df_train.dropna()
    
    if len(df_train) < 10:
        raise ValueError("Dati insufficienti per il training del modello (minimo 10 giorni richiesti)")
    
    X = df_train[['Giorno_Stagione', 'Settimana_Anno', 'Giorno_Settimana', 'Mese']].values
    y = df_train['ADR Bed'].values
    
    models = {}
    
    # 1. Regressione Lineare
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    models['Linear'] = lr_model
    
    # 2. Regressione Polinomiale
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    models['Polynomial'] = (poly_model, poly_features)
    
    # 3. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X, y)
    models['RandomForest'] = rf_model
    
    return models

def generate_forecast_2026(df_historical, models, seasonality, scenario='base'):
    """Genera il forecast per la stagione 2026"""
    
    # Determina il periodo della stagione basandosi sui dati storici
    data_inizio_2025 = df_historical[df_historical['Anno'] == 2025]['Data'].min()
    data_fine_2025 = df_historical[df_historical['Anno'] == 2025]['Data'].max()
    
    # Stima date 2026 (assumendo pattern simile)
    data_inizio_2026 = data_inizio_2025.replace(year=2026)
    durata_stagione = (data_fine_2025 - data_inizio_2025).days
    
    # Genera date per 2026
    date_2026 = pd.date_range(start=data_inizio_2026, periods=durata_stagione + 1, freq='D')
    
    df_2026 = pd.DataFrame({
        'Data': date_2026,
        'Anno': 2026,
        'Mese': date_2026.month,
        'Giorno_Settimana': date_2026.dayofweek,
        'Settimana_Anno': date_2026.isocalendar().week.astype(int),
        'Giorno_Nome': date_2026.day_name(),
        'Mese_Nome': date_2026.month_name()
    })
    
    df_2026['Giorno_Stagione'] = (df_2026['Data'] - data_inizio_2026).dt.days
    
    # Prepara features per la previsione
    X_forecast = df_2026[['Giorno_Stagione', 'Settimana_Anno', 'Giorno_Settimana', 'Mese']]
    
    # Previsioni da diversi modelli
    pred_linear = models['Linear'].predict(X_forecast)
    
    poly_model, poly_features = models['Polynomial']
    X_poly = poly_features.transform(X_forecast)
    pred_poly = poly_model.predict(X_poly)
    
    pred_rf = models['RandomForest'].predict(X_forecast)
    
    # Ensemble: media pesata
    df_2026['ADR_Bed_Linear'] = pred_linear
    df_2026['ADR_Bed_Poly'] = pred_poly
    df_2026['ADR_Bed_RF'] = pred_rf
    
    # Media ensemble con pesi
    weights = {'Linear': 0.2, 'Poly': 0.3, 'RF': 0.5}
    df_2026['ADR_Bed_Ensemble'] = (
        weights['Linear'] * pred_linear +
        weights['Poly'] * pred_poly +
        weights['RF'] * pred_rf
    )
    
    # Applica indice di stagionalità
    df_2026 = df_2026.merge(seasonality[['Settimana_Anno', 'Indice_Stagionalita']], 
                             on='Settimana_Anno', how='left')
    df_2026['Indice_Stagionalita'].fillna(1.0, inplace=True)
    
    df_2026['ADR_Bed_Seasonal'] = df_2026['ADR_Bed_Ensemble'] * df_2026['Indice_Stagionalita']
    
    # Applica scenari
    scenario_adjustments = {
        'pessimistico': 0.90,
        'base': 1.00,
        'ottimistico': 1.10,
        'molto_ottimistico': 1.20
    }
    
    adjustment = scenario_adjustments.get(scenario, 1.0)
    df_2026['ADR_Bed_Forecast'] = df_2026['ADR_Bed_Seasonal'] * adjustment
    
    # Limiti realistici (basati sui dati storici)
    adr_min = df_historical['ADR Bed'].quantile(0.05)
    adr_max = df_historical['ADR Bed'].quantile(0.95)
    df_2026['ADR_Bed_Forecast'] = df_2026['ADR_Bed_Forecast'].clip(lower=adr_min, upper=adr_max)
    
    return df_2026

def calculate_forecast_metrics(df_forecast, df_historical):
    """Calcola metriche chiave del forecast"""
    
    metrics = {
        'ADR_Medio_Forecast': df_forecast['ADR_Bed_Forecast'].mean(),
        'ADR_Medio_2025': df_historical[df_historical['Anno'] == 2025]['ADR Bed'].mean(),
        'ADR_Medio_2024': df_historical[df_historical['Anno'] == 2024]['ADR Bed'].mean(),
        'ADR_Medio_2023': df_historical[df_historical['Anno'] == 2023]['ADR Bed'].mean(),
        'ADR_Min_Forecast': df_forecast['ADR_Bed_Forecast'].min(),
        'ADR_Max_Forecast': df_forecast['ADR_Bed_Forecast'].max(),
        'Giorni_Totali': len(df_forecast)
    }
    
    # Calcola variazioni percentuali
    metrics['Variazione_vs_2025'] = (
        (metrics['ADR_Medio_Forecast'] - metrics['ADR_Medio_2025']) / metrics['ADR_Medio_2025'] * 100
    )
    
    return metrics

# ===============================
# APPLICAZIONE PRINCIPALE
# ===============================

def main():
    
    # Header
    st.markdown('<h1 class="main-header">🏨 VOI Alimini Resort</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Forecasting ADR BED 2026 - Segmenti Diretti</p>', unsafe_allow_html=True)
    
    # File Upload Section
    st.sidebar.header("📁 Caricamento Dati")
    
    with st.sidebar.expander("Carica File Storici", expanded=True):
        uploaded_file_2023 = st.file_uploader(
            "Dati Stagione 2023 (Excel)",
            type=['xlsx', 'xls'],
            key='file_2023',
            help="Carica il file Excel con i dati della stagione 2023"
        )
        
        uploaded_file_2024 = st.file_uploader(
            "Dati Stagione 2024 (Excel)",
            type=['xlsx', 'xls'],
            key='file_2024',
            help="Carica il file Excel con i dati della stagione 2024"
        )
        
        uploaded_file_2025 = st.file_uploader(
            "Dati Stagione 2025 (Excel)",
            type=['xlsx', 'xls'],
            key='file_2025',
            help="Carica il file Excel con i dati della stagione 2025"
        )
    
    # Verifica che tutti i file siano stati caricati
    if not all([uploaded_file_2023, uploaded_file_2024, uploaded_file_2025]):
        st.warning("⚠️ Carica tutti e tre i file Excel (stagioni 2023, 2024, 2025) per procedere con il forecasting.")
        
        st.info("""
        ### 📋 Formato Richiesto
        
        I file Excel devono contenere le seguenti colonne:
        - `Giorno` (formato: "Dom 28/05/2023")
        - `% Occ.` (occupazione percentuale)
        - `Room nights`
        - `Bed nights`
        - `ADR Bed` (Average Daily Rate per posto letto)
        - `RevPar`
        
        ### 📊 Segmenti di Analisi
        Focus su segmenti diretti:
        - SITO WEB
        - WEB PORTALI (OTA)
        - DIRETTI INDIVIDUALI
        """)
        
        st.stop()
    
    # Carica dati
    try:
        with st.spinner('Caricamento e analisi dati storici...'):
            df_historical = load_historical_data(uploaded_file_2023, uploaded_file_2024, uploaded_file_2025)
        
        st.sidebar.success(f"✅ Dati caricati: {len(df_historical)} giorni")
        
        # Mostra info di debug
        with st.sidebar.expander("🔍 Info Dataset", expanded=False):
            st.write(f"**Anni presenti:** {sorted(df_historical['Anno'].unique())}")
            st.write(f"**Periodo:** {df_historical['Data'].min().strftime('%d/%m/%Y')} - {df_historical['Data'].max().strftime('%d/%m/%Y')}")
            st.write(f"**ADR medio:** €{df_historical['ADR Bed'].mean():.2f}")
            st.write(f"**Colonne disponibili:** {len(df_historical.columns)}")
    
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dei dati")
        
        with st.expander("📋 Dettagli Errore (per debug)", expanded=True):
            st.code(str(e))
            
            # Prova a dare più informazioni
            try:
                st.write("**Tentativo di lettura file 2023:**")
                df_test = pd.read_excel(uploaded_file_2023)
                st.write(f"- Righe: {len(df_test)}")
                st.write(f"- Colonne: {list(df_test.columns)}")
                st.dataframe(df_test.head(3))
            except Exception as e2:
                st.error(f"Errore lettura file 2023: {str(e2)}")
        
        st.info("""
        ### 📋 Verifica questi punti:
        
        1. **Formato File**: I file devono essere Excel (.xlsx o .xls)
        2. **Colonne Richieste**: 
           - `Giorno` (formato: "Dom 28/05/2023")
           - `ADR Bed` (numero decimale)
           - `% Occ.`, `Room nights`, `Bed nights`, `RevPar`
        3. **Dati Validi**: Almeno 30 giorni con ADR Bed > 0
        4. **Encoding**: Assicurati che i file non siano corrotti
        """)
        
        st.stop()
    
    # Sidebar per controlli
    st.sidebar.header("⚙️ Configurazione Forecast")
    
    scenario = st.sidebar.selectbox(
        "Scenario Previsionale",
        options=['pessimistico', 'base', 'ottimistico', 'molto_ottimistico'],
        index=1,
        help="Seleziona lo scenario per il forecast 2026"
    )
    
    scenario_descriptions = {
        'pessimistico': "📉 -10% rispetto al trend base",
        'base': "📊 Scenario neutrale basato su dati storici",
        'ottimistico': "📈 +10% rispetto al trend base",
        'molto_ottimistico': "🚀 +20% rispetto al trend base"
    }
    
    st.sidebar.info(scenario_descriptions[scenario])
    
    show_models = st.sidebar.checkbox("Mostra confronto modelli", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dati Storici")
    st.sidebar.metric("Stagioni Analizzate", "3 (2023-2025)")
    st.sidebar.metric("Totale Giorni", len(df_historical))
    
    # Tab principale
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast 2026", 
        "📊 Analisi Storica", 
        "🔍 Comparazione Anni",
        "📉 Metriche Dettagliate"
    ])
    
    # Calcola seasonality e modelli
    try:
        with st.spinner('Costruzione modelli predittivi...'):
            seasonality = calculate_seasonality_index(df_historical)
            models = build_forecast_models(df_historical)
            df_forecast = generate_forecast_2026(df_historical, models, seasonality, scenario)
            metrics = calculate_forecast_metrics(df_forecast, df_historical)
    except Exception as e:
        st.error("❌ Errore nella costruzione dei modelli predittivi")
        
        with st.expander("📋 Dettagli Errore", expanded=True):
            st.code(str(e))
            
            st.write("**Colonne disponibili nel dataset:**")
            st.write(list(df_historical.columns))
            
            st.write("**Primi 5 record:**")
            st.dataframe(df_historical.head())
        
        st.stop()
    
    # =============================
    # TAB 1: FORECAST 2026
    # =============================
    with tab1:
        st.header("Previsione ADR BED per Stagione 2026")
        
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "ADR Medio 2026",
                f"€{metrics['ADR_Medio_Forecast']:.2f}",
                delta=f"{metrics['Variazione_vs_2025']:.1f}% vs 2025"
            )
        
        with col2:
            st.metric(
                "ADR Medio 2025",
                f"€{metrics['ADR_Medio_2025']:.2f}"
            )
        
        with col3:
            st.metric(
                "ADR Min/Max 2026",
                f"€{metrics['ADR_Min_Forecast']:.2f}",
                delta=f"€{metrics['ADR_Max_Forecast']:.2f}"
            )
        
        with col4:
            st.metric(
                "Giorni Stagione",
                metrics['Giorni_Totali']
            )
        
        st.markdown("---")
        
        # Grafico principale forecast
        fig_forecast = go.Figure()
        
        # Dati storici 2025
        df_2025 = df_historical[df_historical['Anno'] == 2025].sort_values('Data')
        fig_forecast.add_trace(go.Scatter(
            x=df_2025['Data'],
            y=df_2025['ADR Bed'],
            mode='lines+markers',
            name='2025 Effettivo',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=4)
        ))
        
        # Forecast 2026
        fig_forecast.add_trace(go.Scatter(
            x=df_forecast['Data'],
            y=df_forecast['ADR_Bed_Forecast'],
            mode='lines+markers',
            name='2026 Forecast',
            line=dict(color='#e74c3c', width=3, dash='solid'),
            marker=dict(size=5)
        ))
        
        # Banda di confidenza (±10%)
        upper_bound = df_forecast['ADR_Bed_Forecast'] * 1.1
        lower_bound = df_forecast['ADR_Bed_Forecast'] * 0.9
        
        fig_forecast.add_trace(go.Scatter(
            x=df_forecast['Data'],
            y=upper_bound,
            mode='lines',
            name='Limite Superiore (+10%)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=df_forecast['Data'],
            y=lower_bound,
            mode='lines',
            name='Limite Inferiore (-10%)',
            line=dict(width=0),
            fillcolor='rgba(231, 76, 60, 0.2)',
            fill='tonexty',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig_forecast.update_layout(
            title="ADR BED: Confronto 2025 vs Forecast 2026",
            xaxis_title="Data",
            yaxis_title="ADR BED (€)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Mostra comparazione modelli se richiesto
        if show_models:
            st.markdown("### Confronto Modelli Predittivi")
            
            fig_models = go.Figure()
            
            fig_models.add_trace(go.Scatter(
                x=df_forecast['Data'],
                y=df_forecast['ADR_Bed_Linear'],
                mode='lines',
                name='Lineare',
                line=dict(dash='dash')
            ))
            
            fig_models.add_trace(go.Scatter(
                x=df_forecast['Data'],
                y=df_forecast['ADR_Bed_Poly'],
                mode='lines',
                name='Polinomiale',
                line=dict(dash='dot')
            ))
            
            fig_models.add_trace(go.Scatter(
                x=df_forecast['Data'],
                y=df_forecast['ADR_Bed_RF'],
                mode='lines',
                name='Random Forest',
                line=dict(dash='dashdot')
            ))
            
            fig_models.add_trace(go.Scatter(
                x=df_forecast['Data'],
                y=df_forecast['ADR_Bed_Forecast'],
                mode='lines',
                name='Ensemble (Finale)',
                line=dict(width=3, color='red')
            ))
            
            fig_models.update_layout(
                title="Comparazione Modelli di Forecasting",
                xaxis_title="Data",
                yaxis_title="ADR BED (€)",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_models, use_container_width=True)
        
        # Tabella mensile
        st.markdown("### 📅 Forecast Mensile 2026")
        
        df_monthly = df_forecast.groupby('Mese_Nome').agg({
            'ADR_Bed_Forecast': 'mean',
            'Data': 'count'
        }).reset_index()
        df_monthly.columns = ['Mese', 'ADR Medio', 'Giorni']
        df_monthly['ADR Medio'] = df_monthly['ADR Medio'].round(2)
        
        st.dataframe(df_monthly, use_container_width=True)
    
    # =============================
    # TAB 2: ANALISI STORICA
    # =============================
    with tab2:
        st.header("Analisi Dati Storici 2023-2025")
        
        # Grafico trend storico
        fig_historical = go.Figure()
        
        for anno in [2023, 2024, 2025]:
            df_anno = df_historical[df_historical['Anno'] == anno].sort_values('Data')
            fig_historical.add_trace(go.Scatter(
                x=df_anno['Giorno_Stagione'],
                y=df_anno['ADR Bed'],
                mode='lines+markers',
                name=f'Stagione {anno}',
                marker=dict(size=4)
            ))
        
        fig_historical.update_layout(
            title="ADR BED: Confronto Stagioni Storiche",
            xaxis_title="Giorno della Stagione",
            yaxis_title="ADR BED (€)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_historical, use_container_width=True)
        
        # Statistiche per anno
        st.markdown("### 📊 Statistiche per Anno")
        
        stats_by_year = df_historical.groupby('Anno')['ADR Bed'].agg([
            ('Media', 'mean'),
            ('Mediana', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std Dev', 'std')
        ]).round(2)
        
        st.dataframe(stats_by_year, use_container_width=True)
        
        # Distribuzione ADR
        st.markdown("### 📈 Distribuzione ADR BED")
        
        fig_dist = go.Figure()
        
        for anno in [2023, 2024, 2025]:
            df_anno = df_historical[df_historical['Anno'] == anno]
            fig_dist.add_trace(go.Box(
                y=df_anno['ADR Bed'],
                name=str(anno),
                boxmean='sd'
            ))
        
        fig_dist.update_layout(
            title="Distribuzione ADR BED per Anno",
            yaxis_title="ADR BED (€)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # =============================
    # TAB 3: COMPARAZIONE ANNI
    # =============================
    with tab3:
        st.header("Comparazione Dettagliata tra Anni")
        
        # Selettore mese
        mesi_disponibili = sorted(df_historical['Mese_Nome'].unique())
        mese_selezionato = st.selectbox("Seleziona Mese", mesi_disponibili)
        
        df_mese = df_historical[df_historical['Mese_Nome'] == mese_selezionato]
        
        # Grafico comparativo per mese
        fig_mese = go.Figure()
        
        for anno in [2023, 2024, 2025]:
            df_anno_mese = df_mese[df_mese['Anno'] == anno].sort_values('Data')
            if len(df_anno_mese) > 0:
                fig_mese.add_trace(go.Scatter(
                    x=df_anno_mese['Data'].dt.day,
                    y=df_anno_mese['ADR Bed'],
                    mode='lines+markers',
                    name=f'{anno}',
                    marker=dict(size=6)
                ))
        
        fig_mese.update_layout(
            title=f"ADR BED - {mese_selezionato}: Comparazione 2023-2025",
            xaxis_title="Giorno del Mese",
            yaxis_title="ADR BED (€)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_mese, use_container_width=True)
        
        # Heatmap giorno settimana
        st.markdown("### 📅 Heatmap ADR BED per Giorno della Settimana")
        
        pivot_dow = df_historical.pivot_table(
            values='ADR Bed',
            index='Anno',
            columns='Giorno_Nome',
            aggfunc='mean'
        )
        
        # Ordina i giorni della settimana
        giorni_ordine = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_dow = pivot_dow[[col for col in giorni_ordine if col in pivot_dow.columns]]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_dow.values,
            x=pivot_dow.columns,
            y=pivot_dow.index,
            colorscale='RdYlGn',
            text=pivot_dow.values.round(2),
            texttemplate='€%{text}',
            textfont={"size": 10},
            colorbar=dict(title="ADR BED (€)")
        ))
        
        fig_heatmap.update_layout(
            title="ADR BED Medio per Giorno della Settimana",
            xaxis_title="Giorno della Settimana",
            yaxis_title="Anno",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # =============================
    # TAB 4: METRICHE DETTAGLIATE
    # =============================
    with tab4:
        st.header("Metriche Dettagliate e Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Crescita Anno su Anno")
            
            growth_metrics = pd.DataFrame({
                'Periodo': ['2023→2024', '2024→2025', '2025→2026 (Forecast)'],
                'Variazione %': [
                    ((metrics['ADR_Medio_2024'] - metrics['ADR_Medio_2023']) / metrics['ADR_Medio_2023'] * 100),
                    ((metrics['ADR_Medio_2025'] - metrics['ADR_Medio_2024']) / metrics['ADR_Medio_2024'] * 100),
                    metrics['Variazione_vs_2025']
                ]
            })
            growth_metrics['Variazione %'] = growth_metrics['Variazione %'].round(2)
            
            st.dataframe(growth_metrics, use_container_width=True)
            
            # Grafico crescita
            fig_growth = go.Figure(go.Bar(
                x=growth_metrics['Periodo'],
                y=growth_metrics['Variazione %'],
                text=growth_metrics['Variazione %'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                marker_color=['#3498db', '#3498db', '#e74c3c']
            ))
            
            fig_growth.update_layout(
                title="Crescita Percentuale Anno su Anno",
                yaxis_title="Variazione %",
                height=350,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Indice di Stagionalità")
            
            fig_season = go.Figure()
            
            fig_season.add_trace(go.Scatter(
                x=seasonality['Settimana_Anno'],
                y=seasonality['Indice_Stagionalita'],
                mode='lines+markers',
                name='Indice Stagionalità',
                line=dict(color='#9b59b6', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.2)'
            ))
            
            # Linea di riferimento a 1.0
            fig_season.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                                annotation_text="Media", annotation_position="right")
            
            fig_season.update_layout(
                title="Indice di Stagionalità per Settimana",
                xaxis_title="Settimana dell'Anno",
                yaxis_title="Indice (1.0 = media)",
                height=350,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_season, use_container_width=True)
        
        # Export forecast
        st.markdown("### 💾 Export Dati")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepara CSV forecast
            df_export = df_forecast[['Data', 'Mese_Nome', 'Giorno_Nome', 'ADR_Bed_Forecast']].copy()
            df_export.columns = ['Data', 'Mese', 'Giorno', 'ADR_BED_2026']
            df_export['ADR_BED_2026'] = df_export['ADR_BED_2026'].round(2)
            
            csv_forecast = df_export.to_csv(index=False)
            
            st.download_button(
                label="📥 Scarica Forecast 2026 (CSV)",
                data=csv_forecast,
                file_name=f"voi_alimini_forecast_2026_{scenario}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Prepara CSV storico
            df_export_hist = df_historical[['Data', 'Anno', 'ADR Bed']].copy()
            df_export_hist.columns = ['Data', 'Anno', 'ADR_BED']
            df_export_hist['ADR_BED'] = df_export_hist['ADR_BED'].round(2)
            
            csv_historical = df_export_hist.to_csv(index=False)
            
            st.download_button(
                label="📥 Scarica Storico 2023-2025 (CSV)",
                data=csv_historical,
                file_name="voi_alimini_storico_2023_2025.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>VOI Alimini Resort - Revenue Management Forecasting System</strong></p>
            <p>Modello basato su 3 stagioni storiche (2023-2025) | Segmenti Diretti: SITO WEB, WEB PORTALI (OTA), DIRETTI INDIVIDUALI</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
