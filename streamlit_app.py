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
        
        # Parsing della data - rimuovi il nome del giorno e parsa solo la data
        # Formato: "Dom 28/05/2023" -> "28/05/2023"
        def parse_italian_date(date_str):
            try:
                # Rimuovi il nome del giorno (primi 3-4 caratteri + spazio)
                date_only = date_str.split(' ', 1)[-1] if isinstance(date_str, str) else date_str
                # Parsa la data nel formato gg/mm/aaaa
                return pd.to_datetime(date_only, format='%d/%m/%Y', errors='coerce')
            except:
                return pd.NaT
        
        df_all['Data'] = df_all['Giorno'].apply(parse_italian_date)
        
        # Conta quante date sono valide prima del filtro
        valid_dates_count = df_all['Data'].notna().sum()
        total_rows = len(df_all)
        
        # Rimuovi righe senza data valida
        df_all = df_all.dropna(subset=['Data'])
        
        if len(df_all) == 0:
            raise ValueError(f"Nessuna data valida trovata nei file ({valid_dates_count}/{total_rows} righe avevano date valide). Verifica il formato delle date nel campo 'Giorno'.")
        
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecast 2026", 
        "📊 Analisi Storica", 
        "🔍 Comparazione Anni",
        "📅 Analisi Mensile",
        "📉 Metriche & Export"
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
        
        # Analisi mensile dettagliata
        st.markdown("### 📅 Analisi Mensile Dettagliata 2026")
        
        # Calcola metriche mensili per forecast
        df_forecast['Mese_Num'] = df_forecast['Data'].dt.month
        monthly_forecast = df_forecast.groupby(['Mese_Num', 'Mese_Nome']).agg({
            'ADR_Bed_Forecast': ['mean', 'min', 'max', 'std'],
            'Data': 'count'
        }).reset_index()
        
        monthly_forecast.columns = ['Mese_Num', 'Mese', 'ADR_Medio', 'ADR_Min', 'ADR_Max', 'ADR_StdDev', 'Giorni']
        
        # Calcola metriche mensili per dati storici
        df_historical['Mese_Num'] = df_historical['Data'].dt.month
        
        monthly_2023 = df_historical[df_historical['Anno'] == 2023].groupby('Mese_Num')['ADR Bed'].mean()
        monthly_2024 = df_historical[df_historical['Anno'] == 2024].groupby('Mese_Num')['ADR Bed'].mean()
        monthly_2025 = df_historical[df_historical['Anno'] == 2025].groupby('Mese_Num')['ADR Bed'].mean()
        
        # Aggiungi confronto con anni precedenti
        monthly_forecast['ADR_2025'] = monthly_forecast['Mese_Num'].map(monthly_2025)
        monthly_forecast['ADR_2024'] = monthly_forecast['Mese_Num'].map(monthly_2024)
        monthly_forecast['ADR_2023'] = monthly_forecast['Mese_Num'].map(monthly_2023)
        
        # Calcola variazioni
        monthly_forecast['Var_vs_2025_%'] = (
            (monthly_forecast['ADR_Medio'] - monthly_forecast['ADR_2025']) / monthly_forecast['ADR_2025'] * 100
        ).round(2)
        
        # Formatta i valori
        monthly_forecast['ADR_Medio'] = monthly_forecast['ADR_Medio'].round(2)
        monthly_forecast['ADR_Min'] = monthly_forecast['ADR_Min'].round(2)
        monthly_forecast['ADR_Max'] = monthly_forecast['ADR_Max'].round(2)
        monthly_forecast['ADR_StdDev'] = monthly_forecast['ADR_StdDev'].round(2)
        monthly_forecast['ADR_2025'] = monthly_forecast['ADR_2025'].round(2)
        monthly_forecast['ADR_2024'] = monthly_forecast['ADR_2024'].round(2)
        monthly_forecast['ADR_2023'] = monthly_forecast['ADR_2023'].round(2)
        
        # Tabella con tutte le informazioni
        display_df = monthly_forecast[['Mese', 'Giorni', 'ADR_Medio', 'ADR_Min', 'ADR_Max', 
                                       'ADR_2025', 'ADR_2024', 'ADR_2023', 'Var_vs_2025_%']].copy()
        
        # Formatta e colora le celle basandosi sulla variazione
        def color_variation(val):
            if pd.isna(val):
                return ''
            if val > 5:
                return 'background-color: #d4edda; color: #155724'  # Verde scuro
            elif val > 0:
                return 'background-color: #d1ecf1; color: #0c5460'  # Azzurro
            elif val > -5:
                return 'background-color: #fff3cd; color: #856404'  # Giallo
            else:
                return 'background-color: #f8d7da; color: #721c24'  # Rosso
        
        styled_df = display_df.style.format({
            'ADR_Medio': '€{:.2f}',
            'ADR_Min': '€{:.2f}',
            'ADR_Max': '€{:.2f}',
            'ADR_2025': '€{:.2f}',
            'ADR_2024': '€{:.2f}',
            'ADR_2023': '€{:.2f}',
            'Var_vs_2025_%': '{:+.2f}%'
        }).applymap(color_variation, subset=['Var_vs_2025_%'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Grafico comparativo mensile
        st.markdown("#### 📊 Confronto ADR Mensile: 2023-2026")
        
        fig_monthly_comparison = go.Figure()
        
        # Ordina i mesi
        monthly_plot = monthly_forecast.sort_values('Mese_Num')
        
        # 2023
        fig_monthly_comparison.add_trace(go.Scatter(
            x=monthly_plot['Mese'],
            y=monthly_plot['ADR_2023'],
            mode='lines+markers',
            name='2023',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8)
        ))
        
        # 2024
        fig_monthly_comparison.add_trace(go.Scatter(
            x=monthly_plot['Mese'],
            y=monthly_plot['ADR_2024'],
            mode='lines+markers',
            name='2024',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=8)
        ))
        
        # 2025
        fig_monthly_comparison.add_trace(go.Scatter(
            x=monthly_plot['Mese'],
            y=monthly_plot['ADR_2025'],
            mode='lines+markers',
            name='2025',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=8)
        ))
        
        # 2026 Forecast
        fig_monthly_comparison.add_trace(go.Scatter(
            x=monthly_plot['Mese'],
            y=monthly_plot['ADR_Medio'],
            mode='lines+markers',
            name='2026 Forecast',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=10, symbol='star')
        ))
        
        # Range min-max 2026
        fig_monthly_comparison.add_trace(go.Scatter(
            x=monthly_plot['Mese'],
            y=monthly_plot['ADR_Max'],
            mode='lines',
            name='Range 2026',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_monthly_comparison.add_trace(go.Scatter(
            x=monthly_plot['Mese'],
            y=monthly_plot['ADR_Min'],
            mode='lines',
            name='Range Min-Max 2026',
            line=dict(width=0),
            fillcolor='rgba(231, 76, 60, 0.15)',
            fill='tonexty',
            showlegend=True
        ))
        
        fig_monthly_comparison.update_layout(
            title="Evoluzione ADR BED Mensile: Storico vs Forecast",
            xaxis_title="Mese",
            yaxis_title="ADR BED (€)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_monthly_comparison, use_container_width=True)
        
        # Grafico variazioni percentuali
        st.markdown("#### 📈 Variazioni Percentuali vs 2025")
        
        fig_variations = go.Figure()
        
        colors = ['#e74c3c' if x >= 0 else '#3498db' for x in monthly_plot['Var_vs_2025_%']]
        
        fig_variations.add_trace(go.Bar(
            x=monthly_plot['Mese'],
            y=monthly_plot['Var_vs_2025_%'],
            text=monthly_plot['Var_vs_2025_%'].apply(lambda x: f"{x:+.1f}%"),
            textposition='outside',
            marker_color=colors,
            name='Variazione %'
        ))
        
        fig_variations.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        
        fig_variations.update_layout(
            title="Crescita/Decrescita Mensile 2026 vs 2025",
            xaxis_title="Mese",
            yaxis_title="Variazione %",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_variations, use_container_width=True)
        
        # Insights mensili
        st.markdown("#### 💡 Insights Mensili")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_month = monthly_plot.loc[monthly_plot['ADR_Medio'].idxmax()]
            st.metric(
                "🏆 Mese Migliore",
                best_month['Mese'],
                f"€{best_month['ADR_Medio']:.2f}"
            )
        
        with col2:
            worst_month = monthly_plot.loc[monthly_plot['ADR_Medio'].idxmin()]
            st.metric(
                "📉 Mese Più Debole",
                worst_month['Mese'],
                f"€{worst_month['ADR_Medio']:.2f}"
            )
        
        with col3:
            highest_growth = monthly_plot.loc[monthly_plot['Var_vs_2025_%'].idxmax()]
            st.metric(
                "🚀 Maggior Crescita",
                highest_growth['Mese'],
                f"{highest_growth['Var_vs_2025_%']:+.1f}%"
            )
    
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
    # TAB 4: ANALISI MENSILE
    # =============================
    with tab4:
        st.header("Analisi Mensile Approfondita 2026")
        
        # Prepara dati mensili completi
        df_forecast['Mese_Num'] = df_forecast['Data'].dt.month
        
        # Selettore mese
        mesi_forecast = sorted(df_forecast['Mese_Nome'].unique())
        mese_selezionato = st.selectbox("Seleziona Mese per Analisi Dettagliata", mesi_forecast, key='monthly_selector')
        
        # Filtra dati per il mese selezionato
        df_mese_2026 = df_forecast[df_forecast['Mese_Nome'] == mese_selezionato].copy()
        mese_num = df_mese_2026['Mese_Num'].iloc[0]
        
        # Dati storici per lo stesso mese
        df_mese_storico = df_historical[df_historical['Mese_Num'] == mese_num]
        
        # Metriche del mese
        st.markdown(f"### 📊 Metriche {mese_selezionato} 2026")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        adr_mese_2026 = df_mese_2026['ADR_Bed_Forecast'].mean()
        adr_mese_2025 = df_mese_storico[df_mese_storico['Anno'] == 2025]['ADR Bed'].mean() if len(df_mese_storico[df_mese_storico['Anno'] == 2025]) > 0 else 0
        var_mese = ((adr_mese_2026 - adr_mese_2025) / adr_mese_2025 * 100) if adr_mese_2025 > 0 else 0
        
        with col1:
            st.metric(
                "ADR Medio Mese",
                f"€{adr_mese_2026:.2f}",
                delta=f"{var_mese:+.1f}% vs 2025"
            )
        
        with col2:
            st.metric(
                "ADR Minimo",
                f"€{df_mese_2026['ADR_Bed_Forecast'].min():.2f}"
            )
        
        with col3:
            st.metric(
                "ADR Massimo",
                f"€{df_mese_2026['ADR_Bed_Forecast'].max():.2f}"
            )
        
        with col4:
            st.metric(
                "Giorni nel Mese",
                len(df_mese_2026)
            )
        
        with col5:
            # Assumendo 308 camere totali (come da VOI Alimini)
            camere_totali = st.number_input("Camere Totali", min_value=1, value=308, key='camere_totali')
            posti_letto_stimati = camere_totali * 2.5  # Media posti per camera
            st.metric(
                "Posti Letto",
                f"{int(posti_letto_stimati)}"
            )
        
        st.markdown("---")
        
        # Revenue Projection
        st.markdown(f"### 💰 Proiezione Revenue {mese_selezionato} 2026")
        
        col1, col2 = st.columns(2)
        
        with col1:
            occupancy_assumption = st.slider(
                "Occupazione Stimata (%)",
                min_value=10,
                max_value=100,
                value=75,
                step=5,
                help="Imposta l'occupazione stimata per calcolare il revenue potenziale"
            )
        
        with col2:
            pax_per_camera = st.slider(
                "Pax per Camera",
                min_value=1.5,
                max_value=3.0,
                value=2.2,
                step=0.1
            )
        
        # Calcoli revenue
        room_nights_mese = (camere_totali * len(df_mese_2026) * occupancy_assumption / 100)
        bed_nights_mese = room_nights_mese * pax_per_camera
        revenue_mese = bed_nights_mese * adr_mese_2026
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Room Nights Previsti",
                f"{room_nights_mese:,.0f}",
                help="Camere x Giorni x Occupazione%"
            )
        
        with col2:
            st.metric(
                "Bed Nights Previsti",
                f"{bed_nights_mese:,.0f}",
                help="Room Nights x Pax per Camera"
            )
        
        with col3:
            st.metric(
                "Revenue Stimato",
                f"€{revenue_mese:,.0f}",
                help="Bed Nights x ADR BED"
            )
        
        st.markdown("---")
        
        # Grafico giornaliero del mese
        st.markdown(f"### 📅 Andamento Giornaliero {mese_selezionato}")
        
        fig_daily = go.Figure()
        
        # 2026 Forecast
        df_mese_2026_sorted = df_mese_2026.sort_values('Data')
        fig_daily.add_trace(go.Scatter(
            x=df_mese_2026_sorted['Data'].dt.day,
            y=df_mese_2026_sorted['ADR_Bed_Forecast'],
            mode='lines+markers',
            name='2026 Forecast',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8),
            text=df_mese_2026_sorted['Giorno_Nome'],
            hovertemplate='<b>Giorno %{x}</b><br>%{text}<br>ADR: €%{y:.2f}<extra></extra>'
        ))
        
        # Aggiungi dati storici se disponibili
        for anno, colore in [(2025, '#2ecc71'), (2024, '#9b59b6'), (2023, '#3498db')]:
            df_anno_mese = df_mese_storico[df_mese_storico['Anno'] == anno].sort_values('Data')
            if len(df_anno_mese) > 0:
                fig_daily.add_trace(go.Scatter(
                    x=df_anno_mese['Data'].dt.day,
                    y=df_anno_mese['ADR Bed'],
                    mode='lines+markers',
                    name=f'{anno}',
                    line=dict(color=colore, width=2, dash='dot'),
                    marker=dict(size=6),
                    opacity=0.7
                ))
        
        # Evidenzia weekend
        weekend_days = df_mese_2026_sorted[df_mese_2026_sorted['Giorno_Settimana'].isin([5, 6])]['Data'].dt.day
        for day in weekend_days:
            fig_daily.add_vrect(
                x0=day-0.5, x1=day+0.5,
                fillcolor="lightblue", opacity=0.1,
                layer="below", line_width=0
            )
        
        fig_daily.update_layout(
            title=f"ADR BED Giornaliero - {mese_selezionato}",
            xaxis_title="Giorno del Mese",
            yaxis_title="ADR BED (€)",
            hovermode='x unified',
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Analisi per giorno della settimana
        st.markdown(f"### 📊 Analisi per Giorno della Settimana - {mese_selezionato}")
        
        df_mese_2026['Giorno_Nome_IT'] = df_mese_2026['Giorno_Settimana'].map({
            0: 'Lunedì', 1: 'Martedì', 2: 'Mercoledì', 3: 'Giovedì',
            4: 'Venerdì', 5: 'Sabato', 6: 'Domenica'
        })
        
        dow_analysis = df_mese_2026.groupby('Giorno_Nome_IT')['ADR_Bed_Forecast'].agg(['mean', 'count']).reset_index()
        dow_analysis.columns = ['Giorno', 'ADR Medio', 'Occorrenze']
        dow_analysis['ADR Medio'] = dow_analysis['ADR Medio'].round(2)
        
        # Ordina giorni settimana
        giorni_ordine = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
        dow_analysis['Giorno'] = pd.Categorical(dow_analysis['Giorno'], categories=giorni_ordine, ordered=True)
        dow_analysis = dow_analysis.sort_values('Giorno')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_dow = go.Figure()
            
            colors_dow = ['#3498db' if g in ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì'] 
                         else '#e74c3c' for g in dow_analysis['Giorno']]
            
            fig_dow.add_trace(go.Bar(
                x=dow_analysis['Giorno'],
                y=dow_analysis['ADR Medio'],
                text=dow_analysis['ADR Medio'].apply(lambda x: f"€{x:.0f}"),
                textposition='outside',
                marker_color=colors_dow,
                name='ADR Medio'
            ))
            
            fig_dow.update_layout(
                title="ADR Medio per Giorno della Settimana",
                xaxis_title="Giorno",
                yaxis_title="ADR BED (€)",
                height=350,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            st.markdown("#### Statistiche Settimanali")
            
            adr_weekday = dow_analysis[dow_analysis['Giorno'].isin(['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì'])]['ADR Medio'].mean()
            adr_weekend = dow_analysis[dow_analysis['Giorno'].isin(['Sabato', 'Domenica'])]['ADR Medio'].mean()
            premium_weekend = ((adr_weekend - adr_weekday) / adr_weekday * 100) if adr_weekday > 0 else 0
            
            st.metric("ADR Infrasettimanale", f"€{adr_weekday:.2f}")
            st.metric("ADR Weekend", f"€{adr_weekend:.2f}")
            st.metric("Premium Weekend", f"+{premium_weekend:.1f}%")
        
        # Tabella dettagliata giornaliera
        st.markdown(f"### 📋 Dettaglio Giornaliero {mese_selezionato} 2026")
        
        # Ordina i dati per data
        df_mese_2026_sorted = df_mese_2026.sort_values('Data').copy()
        
        df_detail = df_mese_2026_sorted[['Data', 'Giorno_Nome_IT', 'ADR_Bed_Forecast']].copy()
        df_detail['Data'] = df_detail['Data'].dt.strftime('%d/%m/%Y')
        df_detail.columns = ['Data', 'Giorno', 'ADR BED Forecast']
        df_detail['ADR BED Forecast'] = df_detail['ADR BED Forecast'].round(2)
        
        # Aggiungi revenue stimato giornaliero
        bed_nights_giornalieri = (camere_totali * pax_per_camera * occupancy_assumption / 100)
        df_detail['Revenue Stimato'] = (df_detail['ADR BED Forecast'] * bed_nights_giornalieri).round(0)
        
        st.dataframe(
            df_detail.style.format({
                'ADR BED Forecast': '€{:.2f}',
                'Revenue Stimato': '€{:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Export mensile
        csv_monthly = df_detail.to_csv(index=False)
        st.download_button(
            label=f"📥 Scarica Dettaglio {mese_selezionato} 2026 (CSV)",
            data=csv_monthly,
            file_name=f"voi_alimini_{mese_selezionato.lower()}_2026_forecast.csv",
            mime="text/csv"
        )
    
    # =============================
    # TAB 5: METRICHE DETTAGLIATE
    # =============================
    with tab5:
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
