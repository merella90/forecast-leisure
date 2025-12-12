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

# ===============================
# BOOKING CURVE FUNCTIONS
# ===============================

@st.cache_data
def load_snapshots_2025():
    """Carica tutte le snapshot storiche 2025 da GitHub"""
    
    snapshots = []
    snapshot_dates = [
        ('snapshot_01_nov_2024.xlsx', '2024-11-01'),
        ('snapshot_02_dic_2024.xlsx', '2024-12-01'),
        ('snapshot_03_gen_2025.xlsx', '2025-01-01'),
        ('snapshot_04_feb_2025.xlsx', '2025-02-01'),
        ('snapshot_05_mar_2025.xlsx', '2025-03-01'),
        ('snapshot_06_apr_2025.xlsx', '2025-04-01'),
        ('snapshot_07_mag_2025.xlsx', '2025-05-01'),
        ('snapshot_08_giu_2025.xlsx', '2025-06-01'),
        ('snapshot_09_lug_2025.xlsx', '2025-07-01'),
        ('snapshot_10_ago_2025.xlsx', '2025-08-01'),
        ('snapshot_11_set_2025.xlsx', '2025-09-01'),
        ('snapshot_12_finale_2025.xlsx', '2025-10-01'),
    ]
    
    base_path = 'snapshots_2025/'
    
    for filename, snapshot_date in snapshot_dates:
        try:
            filepath = base_path + filename
            df = pd.read_excel(filepath)
            
            # Parse dates
            def parse_date(date_str):
                try:
                    date_only = date_str.split(' ', 1)[-1] if isinstance(date_str, str) else date_str
                    return pd.to_datetime(date_only, format='%d/%m/%Y', errors='coerce')
                except:
                    return pd.NaT
            
            df['Data'] = df['Giorno'].apply(parse_date)
            df = df.dropna(subset=['Data'])
            df = df[df['ADR Bed'] > 0].copy()
            
            if len(df) > 0:
                df['Snapshot_Date'] = pd.to_datetime(snapshot_date)
                df['Snapshot_Label'] = pd.to_datetime(snapshot_date).strftime('%b %Y')
                snapshots.append(df)
        
        except Exception as e:
            st.warning(f"Impossibile caricare {filename}: {str(e)}")
            continue
    
    if len(snapshots) > 0:
        return pd.concat(snapshots, ignore_index=True)
    else:
        return None

def load_snapshot_2026(uploaded_file):
    """Carica e processa lo snapshot OTB 2026 corrente"""
    
    try:
        df = pd.read_excel(uploaded_file)
        
        # Parse dates
        def parse_date(date_str):
            try:
                date_only = date_str.split(' ', 1)[-1] if isinstance(date_str, str) else date_str
                return pd.to_datetime(date_only, format='%d/%m/%Y', errors='coerce')
            except:
                return pd.NaT
        
        df['Data'] = df['Giorno'].apply(parse_date)
        df = df.dropna(subset=['Data'])
        df = df[df['ADR Bed'] > 0].copy()
        
        if len(df) > 0:
            df['Snapshot_Date'] = pd.to_datetime('today')
            df['Snapshot_Label'] = pd.to_datetime('today').strftime('%b %Y')
            return df
        else:
            return None
    
    except Exception as e:
        st.error(f"Errore nel caricamento dello snapshot 2026: {str(e)}")
        return None

def calculate_booking_window(snapshot_date, stay_date):
    """Calcola i giorni di booking window (quanti giorni prima della data di soggiorno)"""
    return (stay_date - snapshot_date).days

def compare_booking_curves(df_snapshots_2025, df_snapshot_2026, force_date=None, debug=False):
    """Confronta le booking curves 2025 vs 2026"""
    
    # Data snapshot 2026
    snapshot_date_2026 = df_snapshot_2026['Snapshot_Date'].iloc[0]
    
    # Se viene fornita una data specifica da usare, cerca quella esatta
    if force_date is not None:
        if debug:
            st.write(f"🔍 force_date fornita: **{force_date.strftime('%d/%m/%Y')}**")
        
        # Cerca snapshot 2025 alla data esatta
        matching_snapshots = df_snapshots_2025[
            df_snapshots_2025['Snapshot_Date'].dt.date == force_date.date()
        ]
        
        if debug:
            st.write(f"🔍 Snapshot trovate per {force_date.date()}: **{len(matching_snapshots)}**")
            if len(matching_snapshots) > 0:
                st.write(f"🔍 Room Nights nella snapshot: **{matching_snapshots['Room nights'].sum():.0f}**")
        
        if len(matching_snapshots) > 0:
            closest_snapshot = force_date
            df_2025_comparable = matching_snapshots.copy()
            if debug:
                st.success(f"✅ Usando snapshot ESATTA del {closest_snapshot.strftime('%d/%m/%Y')}")
        else:
            # Fallback: cerca la più vicina
            if debug:
                st.warning("⚠️ Snapshot esatta non trovata, usando fallback")
            snapshots_2025_unique = df_snapshots_2025.groupby('Snapshot_Date').size().reset_index()
            snapshots_2025_unique['diff'] = abs(
                (snapshots_2025_unique['Snapshot_Date'] - force_date).dt.days
            )
            closest_snapshot = snapshots_2025_unique.loc[snapshots_2025_unique['diff'].idxmin(), 'Snapshot_Date']
            df_2025_comparable = df_snapshots_2025[df_snapshots_2025['Snapshot_Date'] == closest_snapshot].copy()
            if debug:
                st.info(f"ℹ️ Usando snapshot più vicina del {closest_snapshot.strftime('%d/%m/%Y')}")
    else:
        # Trova snapshot 2025 più vicina (logica originale)
        if debug:
            st.info("ℹ️ Nessuna force_date, usando snapshot più vicina")
        snapshots_2025_unique = df_snapshots_2025.groupby('Snapshot_Date').size().reset_index()
        snapshots_2025_unique['diff'] = abs(
            (snapshots_2025_unique['Snapshot_Date'] - snapshot_date_2026.replace(year=2024)).dt.days
        )
        closest_snapshot = snapshots_2025_unique.loc[snapshots_2025_unique['diff'].idxmin(), 'Snapshot_Date']
        df_2025_comparable = df_snapshots_2025[df_snapshots_2025['Snapshot_Date'] == closest_snapshot].copy()
        if debug:
            st.info(f"ℹ️ Snapshot più vicina: {closest_snapshot.strftime('%d/%m/%Y')}")
    
    # Calcola totali per confronto
    comparison = {
        'snapshot_date_2025': closest_snapshot,
        'snapshot_date_2026': snapshot_date_2026,
        'room_nights_2025': df_2025_comparable['Room nights'].sum(),
        'room_nights_2026': df_snapshot_2026['Room nights'].sum(),
        'adr_2025': df_2025_comparable['ADR Bed'].mean(),
        'adr_2026': df_snapshot_2026['ADR Bed'].mean(),
        'revenue_2025': (df_2025_comparable['Room nights'] * df_2025_comparable['ADR Bed']).sum(),
        'revenue_2026': (df_snapshot_2026['Room nights'] * df_snapshot_2026['ADR Bed']).sum(),
    }
    
    # Calcola gap
    comparison['gap_room_nights'] = comparison['room_nights_2026'] - comparison['room_nights_2025']
    comparison['gap_room_nights_pct'] = (comparison['gap_room_nights'] / comparison['room_nights_2025'] * 100) if comparison['room_nights_2025'] > 0 else 0
    comparison['gap_adr'] = comparison['adr_2026'] - comparison['adr_2025']
    comparison['gap_adr_pct'] = (comparison['gap_adr'] / comparison['adr_2025'] * 100) if comparison['adr_2025'] > 0 else 0
    comparison['gap_revenue'] = comparison['revenue_2026'] - comparison['revenue_2025']
    comparison['gap_revenue_pct'] = (comparison['gap_revenue'] / comparison['revenue_2025'] * 100) if comparison['revenue_2025'] > 0 else 0
    
    return comparison, df_2025_comparable

def calculate_pickup_rates(df_snapshots_2025):
    """Calcola i pickup rates tra snapshot consecutive"""
    
    snapshots_sorted = sorted(df_snapshots_2025['Snapshot_Date'].unique())
    
    pickup_data = []
    
    for i in range(len(snapshots_sorted) - 1):
        snap_current = snapshots_sorted[i]
        snap_next = snapshots_sorted[i + 1]
        
        df_current = df_snapshots_2025[df_snapshots_2025['Snapshot_Date'] == snap_current]
        df_next = df_snapshots_2025[df_snapshots_2025['Snapshot_Date'] == snap_next]
        
        rn_current = df_current['Room nights'].sum()
        rn_next = df_next['Room nights'].sum()
        pickup = rn_next - rn_current
        
        days_between = (snap_next - snap_current).days
        pickup_per_day = pickup / days_between if days_between > 0 else 0
        
        pickup_data.append({
            'from_date': snap_current,
            'to_date': snap_next,
            'from_label': snap_current.strftime('%b %Y'),
            'to_label': snap_next.strftime('%b %Y'),
            'days_between': days_between,
            'room_nights_start': rn_current,
            'room_nights_end': rn_next,
            'pickup_total': pickup,
            'pickup_per_day': pickup_per_day
        })
    
    return pd.DataFrame(pickup_data)

def generate_rm_suggestions(comparison, monthly_gap, pickup_forecast):
    """Genera suggerimenti di revenue management basati sui dati"""
    
    suggestions = []
    
    # Alert generale room nights
    gap_pct = comparison['gap_room_nights_pct']
    if gap_pct < -15:
        suggestions.append({
            'type': 'critical',
            'icon': '🔴',
            'title': 'CRITICO: Gap Room Nights Significativo',
            'message': f"Sei {abs(gap_pct):.1f}% indietro rispetto al 2025 ({abs(comparison['gap_room_nights']):.0f} RN). Azione immediata richiesta.",
            'actions': [
                'Attiva campagne promozionali early booking',
                'Considera riduzione tariffe per periodi deboli',
                'Aumenta visibilità su OTA con deals speciali',
                'Valuta flash sale per stimolare le prenotazioni'
            ]
        })
    elif gap_pct < -5:
        suggestions.append({
            'type': 'warning',
            'icon': '🟡',
            'title': 'Attenzione: Booking Pace Lento',
            'message': f"Sei {abs(gap_pct):.1f}% indietro rispetto al 2025 ({abs(comparison['gap_room_nights']):.0f} RN).",
            'actions': [
                'Monitora quotidianamente l\'evoluzione',
                'Prepara campagne promozionali di riserva',
                'Verifica posizionamento su OTA',
                'Considera apertura early booking per periodi specifici'
            ]
        })
    elif gap_pct > 5:
        suggestions.append({
            'type': 'success',
            'icon': '🟢',
            'title': 'Ottimo: Booking Pace Forte',
            'message': f"Sei {gap_pct:.1f}% avanti rispetto al 2025 (+{comparison['gap_room_nights']:.0f} RN)!",
            'actions': [
                'Considera aumento tariffe per periodi ad alta domanda',
                'Chiudi canali low-rate per proteggere ADR',
                'Implementa strategia di yield management aggressiva',
                'Valuta upgrade a camere premium'
            ]
        })
    
    # Alert ADR
    adr_gap_pct = comparison['gap_adr_pct']
    if adr_gap_pct > 5:
        suggestions.append({
            'type': 'success',
            'icon': '💰',
            'title': 'Eccellente: ADR in Crescita',
            'message': f"ADR +{adr_gap_pct:.1f}% rispetto al 2025 (€{comparison['gap_adr']:.2f}).",
            'actions': [
                'Mantieni strategia pricing attuale',
                'Continua a proteggere il posizionamento premium'
            ]
        })
    elif adr_gap_pct < -5:
        suggestions.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': 'Attenzione: ADR in Calo',
            'message': f"ADR {adr_gap_pct:.1f}% rispetto al 2025 (€{comparison['gap_adr']:.2f}).",
            'actions': [
                'Rivedi strategia pricing',
                'Valuta chiusura canali discount',
                'Aumenta valore percepito con pacchetti inclusi'
            ]
        })
    
    # Suggerimenti per mesi specifici
    if monthly_gap is not None and len(monthly_gap) > 0:
        weak_months = monthly_gap[monthly_gap['gap_pct'] < -20]
        if len(weak_months) > 0:
            mesi_critici = ', '.join(weak_months['mese'].tolist())
            suggestions.append({
                'type': 'critical',
                'icon': '📅',
                'title': f'Mesi Critici: {mesi_critici}',
                'message': 'Alcuni mesi hanno gap superiori a -20%',
                'actions': [
                    f'Focus immediato su {mesi_critici}',
                    'Crea pacchetti specifici per questi periodi',
                    'Intensifica marketing per queste date'
                ]
            })
    
    return suggestions

def create_hybrid_forecast(df_forecast_ml, df_snapshot_2026, df_snapshots_2025, df_historical):
    """
    Crea forecast ibrido che combina:
    1. Dati OTB 2026 reali (dove disponibili)
    2. Forecast ML aggiustato con pickup rate reale
    3. Correzione ADR basata su trend reale
    """
    
    # Copia il forecast ML di base
    df_hybrid = df_forecast_ml.copy()
    
    # Parse dates dello snapshot 2026
    df_otb_2026 = df_snapshot_2026.copy()
    
    # Identifica date con OTB reale
    date_otb = set(df_otb_2026['Data'].dt.date)
    
    # Per le date con OTB reale, usa i dati reali
    for idx, row in df_hybrid.iterrows():
        date_forecast = row['Data'].date()
        
        if date_forecast in date_otb:
            # Trova il record OTB corrispondente
            otb_row = df_otb_2026[df_otb_2026['Data'].dt.date == date_forecast]
            
            if len(otb_row) > 0:
                otb_row = otb_row.iloc[0]
                
                # Usa ADR reale se disponibile e ragionevole
                if pd.notna(otb_row['ADR Bed']) and otb_row['ADR Bed'] > 0:
                    df_hybrid.at[idx, 'ADR_Bed_Forecast'] = otb_row['ADR Bed']
                    df_hybrid.at[idx, 'Source'] = 'OTB_Real'
                
                # Usa Room Nights reali se disponibili
                if 'Room nights' in otb_row and pd.notna(otb_row['Room nights']):
                    df_hybrid.at[idx, 'Room_Nights_Real'] = otb_row['Room nights']
    
    # Calcola fattore di aggiustamento ADR basato su trend reale
    adr_otb_mean = df_otb_2026['ADR Bed'].mean()
    adr_forecast_comparable = df_hybrid[df_hybrid['Data'].isin(df_otb_2026['Data'])]['ADR_Bed_Forecast'].mean()
    
    if adr_forecast_comparable > 0:
        adr_adjustment_factor = adr_otb_mean / adr_forecast_comparable
    else:
        adr_adjustment_factor = 1.0
    
    # Applica aggiustamento alle date senza OTB (solo se fattore è ragionevole)
    if 0.8 <= adr_adjustment_factor <= 1.2:
        mask_no_otb = ~df_hybrid['Data'].dt.date.isin(date_otb)
        df_hybrid.loc[mask_no_otb, 'ADR_Bed_Forecast'] *= adr_adjustment_factor
        df_hybrid.loc[mask_no_otb, 'Source'] = 'ML_Adjusted'
    
    # Aggiungi flag per identificare source
    df_hybrid['Source'] = df_hybrid.get('Source', 'ML_Original')
    
    # Ricalcola metriche derivate
    if 'Occupazione_Forecast' in df_hybrid.columns:
        df_hybrid['Revenue_Forecast'] = (
            df_hybrid['Room_Nights_Forecast'] * 
            df_hybrid['ADR_Bed_Forecast'] * 
            df_hybrid.get('Room_Nights_Real', 1.0).fillna(1.0)
        )
    
    return df_hybrid, adr_adjustment_factor

# ===============================
# ORIGINAL FUNCTIONS
# ===============================

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
    
    # Aggiungi metriche Room Nights se disponibili
    if 'Room_Nights_Forecast' in df_forecast.columns:
        metrics['Room_Nights_Totali'] = df_forecast['Room_Nights_Forecast'].sum()
        metrics['Room_Nights_Medi_Giorno'] = df_forecast['Room_Nights_Forecast'].mean()
        metrics['Occupazione_Media'] = df_forecast['Occupazione_Forecast'].mean() * 100
    
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
    
    # Prova a caricare i file storici da GitHub
    try:
        with st.spinner('Caricamento dati storici da GitHub...'):
            df_historical = load_historical_data('data_2023.xlsx', 'data_2024.xlsx', 'data_2025.xlsx')
        
        st.sidebar.success(f"✅ Dati storici caricati: {len(df_historical)} giorni")
        
        # Mostra info di debug
        with st.sidebar.expander("🔍 Info Dataset", expanded=False):
            st.write(f"**Anni presenti:** {sorted(df_historical['Anno'].unique())}")
            st.write(f"**Periodo:** {df_historical['Data'].min().strftime('%d/%m/%Y')} - {df_historical['Data'].max().strftime('%d/%m/%Y')}")
            st.write(f"**ADR medio:** €{df_historical['ADR Bed'].mean():.2f}")
            st.write(f"**Colonne disponibili:** {len(df_historical.columns)}")
        
        files_loaded_from_github = True
    
    except Exception as e:
        files_loaded_from_github = False
        st.sidebar.warning("⚠️ File storici non trovati su GitHub")
        
        with st.sidebar.expander("Carica File Storici Manualmente", expanded=True):
            st.info("I file data_2023.xlsx, data_2024.xlsx, data_2025.xlsx non sono presenti su GitHub. Caricali manualmente.")
            
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
                st.warning("⚠️ Carica tutti e tre i file Excel (stagioni 2023, 2024, 2025) per procedere.")
                
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
            
            # Carica dati dai file uploadati manualmente
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
    
    # NUOVO: Upload Snapshot 2026 OTB
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Snapshot OTB 2026")
    
    uploaded_snapshot_2026 = st.sidebar.file_uploader(
        "**Snapshot OTB 2026 Corrente** (Obbligatorio)",
        type=['xlsx', 'xls'],
        key='sidebar_snapshot_2026',
        help="Snapshot OTB attuale per stagione 2026 - NECESSARIO per forecast accurato"
    )
    
    # Variabile per snapshot 2025 comparabile (opzionale ma consigliata)
    uploaded_snapshot_2025_comparable = None
    snapshot_2026_date = None
    
    # Se snapshot 2026 è caricato, analizza la data
    if uploaded_snapshot_2026:
        try:
            # Leggi snapshot 2026 per estrarre la data
            df_temp_2026 = pd.read_excel(uploaded_snapshot_2026)
            
            # Reset file pointer dopo lettura
            uploaded_snapshot_2026.seek(0)
            
            # Prova a determinare la data dello snapshot (data odierna come fallback)
            snapshot_2026_date = pd.to_datetime('today')
            
            # Calcola data comparabile 2025 (stesso giorno/mese dell'anno prima)
            snapshot_2025_target_date = snapshot_2026_date.replace(year=2024)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎯 Confronto Ottimale")
            
            st.sidebar.info(f"""
            **Data Snapshot 2026:** {snapshot_2026_date.strftime('%d/%m/%Y')}
            
            **Data ideale snapshot 2025:** {snapshot_2025_target_date.strftime('%d/%m/%Y')}
            
            Per un confronto perfetto allo stesso booking window, carica anche lo snapshot 2025 di quella data.
            """)
            
            uploaded_snapshot_2025_comparable = st.sidebar.file_uploader(
                f"Snapshot 2025 al {snapshot_2025_target_date.strftime('%d/%m/%Y')} (Opzionale)",
                type=['xlsx', 'xls'],
                key='snapshot_2025_comparable',
                help="Se disponibile, permette confronto esatto allo stesso booking window"
            )
            
            if uploaded_snapshot_2025_comparable:
                st.sidebar.success("✅ Confronto perfetto attivato!")
            else:
                st.sidebar.warning("⚠️ Userò snapshot 2025 più vicina disponibile su GitHub")
        
        except Exception as e:
            st.sidebar.error(f"Errore lettura snapshot 2026: {str(e)}")
    
    # NUOVO: Verifica snapshot 2026
    if not uploaded_snapshot_2026:
        st.error("🔴 **SNAPSHOT OTB 2026 OBBLIGATORIA**")
        
        st.warning("""
        ### 📊 Snapshot OTB 2026 Mancante
        
        Per un forecasting accurato è **NECESSARIO** caricare lo snapshot OTB 2026 corrente.
        
        **Perché è obbligatorio?**
        - ✅ Aggiorna il forecast con dati reali di prenotazione
        - ✅ Calcola gap vs anno precedente
        - ✅ Genera suggerimenti Revenue Management
        - ✅ Corregge il modello ML con pickup reale
        
        **Caricalo nella sidebar** → "Snapshot OTB 2026 Corrente"
        """)
        
        st.info("""
        ### 🎯 Cosa Ottieni con lo Snapshot 2026
        
        1. **Forecast Ibrido**: Combina modello ML + dati reali OTB
        2. **Booking Curve Analysis**: Confronto dettagliato 2025 vs 2026
        3. **Gap Analysis**: Room Nights, ADR, Revenue per mese
        4. **RM Suggestions**: Alert e azioni consigliate automatiche
        5. **Pickup Forecast**: Previsione basata su pickup reale
        
        ### 💡 Pro Tip
        
        Se hai anche lo snapshot 2025 **esattamente alla stessa data dell'anno scorso**, 
        caricalo per un confronto perfetto allo stesso booking window!
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
    
    # Configurazione Room Nights
    st.sidebar.markdown("### 🏨 Configurazione Struttura")
    
    camere_totali = st.sidebar.number_input(
        "Numero Camere Totali",
        min_value=100,
        max_value=500,
        value=308,
        step=1,
        help="Numero totale di camere dell'hotel"
    )
    
    st.sidebar.markdown("### 📊 Occupazione Forecast")
    
    occupancy_scenario = st.sidebar.selectbox(
        "Scenario Occupazione",
        options=['conservativo', 'moderato', 'ottimistico'],
        index=1,
        help="Livello di occupazione previsto per il 2026"
    )
    
    occupancy_scenarios = {
        'conservativo': 0.65,  # 65%
        'moderato': 0.75,      # 75%
        'ottimistico': 0.85    # 85%
    }
    
    occupancy_base = occupancy_scenarios[occupancy_scenario]
    
    st.sidebar.info(f"📈 Occupazione base: {occupancy_base*100:.0f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dati Storici")
    st.sidebar.metric("Stagioni Analizzate", "3 (2023-2025)")
    st.sidebar.metric("Totale Giorni", len(df_historical))
    
    # Tab principale
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Forecast 2026", 
        "📊 Analisi Storica", 
        "🔍 Comparazione Anni",
        "📅 Analisi Mensile",
        "📉 Booking Curve & RM",
        "💾 Metriche & Export"
    ])
    
    # Calcola seasonality e modelli
    try:
        with st.spinner('Costruzione modelli predittivi...'):
            seasonality = calculate_seasonality_index(df_historical)
            models = build_forecast_models(df_historical)
            df_forecast_base = generate_forecast_2026(df_historical, models, seasonality, scenario)
        
        # Carica snapshot 2025 e 2026 per forecast ibrido
        with st.spinner('Caricamento snapshot OTB...'):
            df_snapshots_2025 = load_snapshots_2025()
            
            # Variabile per tracciare se abbiamo snapshot comparabile esatta
            has_exact_comparable = False
            exact_comparable_date = None
            
            # Se disponibile, usa snapshot 2025 comparabile caricato dall'utente
            if uploaded_snapshot_2025_comparable:
                df_snapshot_2025_user = load_snapshot_2026(uploaded_snapshot_2025_comparable)  # Riusa la stessa funzione
                if df_snapshot_2025_user is not None:
                    # Calcola data target
                    target_date = snapshot_2026_date.replace(year=2024)
                    
                    # DEBUG: Mostra info
                    st.sidebar.write(f"🔍 DEBUG: Target date = {target_date.strftime('%d/%m/%Y')}")
                    
                    # IMPORTANTE: Rimuovi snapshot GitHub alla stessa data se esiste
                    if df_snapshots_2025 is not None:
                        before_count = len(df_snapshots_2025['Snapshot_Date'].unique())
                        df_snapshots_2025 = df_snapshots_2025[
                            df_snapshots_2025['Snapshot_Date'].dt.date != target_date.date()
                        ]
                        after_count = len(df_snapshots_2025['Snapshot_Date'].unique())
                        st.sidebar.write(f"🔍 DEBUG: Snapshot rimosse: {before_count - after_count}")
                    
                    # Aggiorna la data dello snapshot user
                    df_snapshot_2025_user['Snapshot_Date'] = target_date
                    df_snapshot_2025_user['Snapshot_Label'] = target_date.strftime('%b %Y')
                    
                    # Aggiungi alle snapshot 2025
                    if df_snapshots_2025 is not None:
                        df_snapshots_2025 = pd.concat([df_snapshots_2025, df_snapshot_2025_user], ignore_index=True)
                    else:
                        df_snapshots_2025 = df_snapshot_2025_user
                    
                    has_exact_comparable = True
                    exact_comparable_date = target_date
                    
                    # DEBUG: Mostra tutte le date snapshot disponibili
                    unique_dates = sorted(df_snapshots_2025['Snapshot_Date'].unique())
                    st.sidebar.write(f"🔍 DEBUG: Date snapshot disponibili: {[d.strftime('%d/%m/%Y') for d in unique_dates]}")
                    
                    st.sidebar.success(f"✅ Snapshot 2025 del {target_date.strftime('%d/%m/%Y')} integrata!")
            
            df_snapshot_2026 = load_snapshot_2026(uploaded_snapshot_2026)
        
        if df_snapshot_2026 is None:
            st.error("❌ Errore nel caricamento dello snapshot 2026")
            st.stop()
        
        # Crea forecast ibrido (ML + OTB reale)
        with st.spinner('Creazione forecast ibrido con dati OTB reali...'):
            df_forecast, adr_adjustment = create_hybrid_forecast(
                df_forecast_base, 
                df_snapshot_2026, 
                df_snapshots_2025,
                df_historical
            )
            
            # Aggiungi forecast Room Nights e Occupazione
            # Calcola occupazione con stagionalità
            df_forecast['Occupazione_Forecast'] = occupancy_base * df_forecast['Indice_Stagionalita']
            df_forecast['Occupazione_Forecast'] = df_forecast['Occupazione_Forecast'].clip(upper=0.98)  # Max 98%
            
            # Calcola Room Nights
            df_forecast['Room_Nights_Forecast'] = camere_totali * df_forecast['Occupazione_Forecast']
            
            # Calcola Bed Nights (assumendo 2.2 pax/camera medio)
            pax_per_camera_medio = 2.2
            df_forecast['Bed_Nights_Forecast'] = df_forecast['Room_Nights_Forecast'] * pax_per_camera_medio
            
            # Calcola Revenue
            df_forecast['Revenue_Forecast'] = df_forecast['Bed_Nights_Forecast'] * df_forecast['ADR_Bed_Forecast']
            
            # Calcola RevPAR
            df_forecast['RevPAR_Forecast'] = df_forecast['ADR_Bed_Forecast'] * df_forecast['Occupazione_Forecast']
            
            metrics = calculate_forecast_metrics(df_forecast, df_historical)
        
        # Info forecast ibrido
        st.sidebar.success("✅ Forecast Ibrido Creato")
        
        with st.sidebar.expander("ℹ️ Info Forecast", expanded=False):
            otb_days = (df_forecast['Source'] == 'OTB_Real').sum()
            adjusted_days = (df_forecast['Source'] == 'ML_Adjusted').sum()
            ml_days = (df_forecast['Source'] == 'ML_Original').sum()
            
            st.write(f"**Giorni OTB Reale:** {otb_days}")
            st.write(f"**Giorni ML Aggiustato:** {adjusted_days}")
            st.write(f"**Giorni ML Base:** {ml_days}")
            st.write(f"**Fattore Aggiustamento ADR:** {adr_adjustment:.2%}")
        
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
        
        # Badge Forecast Ibrido
        otb_days = (df_forecast['Source'] == 'OTB_Real').sum()
        
        col_badge1, col_badge2, col_badge3 = st.columns([1, 1, 2])
        with col_badge1:
            st.info(f"🎯 **Forecast Ibrido**")
        with col_badge2:
            st.success(f"✅ {otb_days} giorni OTB reali integrati")
        with col_badge3:
            st.write(f"_Fattore aggiustamento ADR: {adr_adjustment:.1%}_")
        
        st.markdown("---")
        
        # Metriche principali - Prima riga
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
                "Room Nights Totali",
                f"{metrics.get('Room_Nights_Totali', 0):,.0f}",
                help="Totale room nights previsti per la stagione 2026"
            )
        
        with col4:
            st.metric(
                "Occupazione Media",
                f"{metrics.get('Occupazione_Media', 0):.1f}%",
                help="Occupazione media prevista per la stagione 2026"
            )
        
        # Metriche principali - Seconda riga
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue_totale = df_forecast['Revenue_Forecast'].sum()
            st.metric(
                "Revenue Totale 2026",
                f"€{revenue_totale:,.0f}",
                help="Revenue totale previsto per la stagione 2026"
            )
        
        with col2:
            revpar_medio = df_forecast['RevPAR_Forecast'].mean()
            st.metric(
                "RevPAR Medio",
                f"€{revpar_medio:.2f}",
                help="Revenue Per Available Room medio"
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
        
        # Grafici Room Nights e Revenue
        st.markdown("### 📊 Room Nights e Revenue 2026")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_room_nights = go.Figure()
            
            fig_room_nights.add_trace(go.Scatter(
                x=df_forecast['Data'],
                y=df_forecast['Room_Nights_Forecast'],
                mode='lines',
                name='Room Nights',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig_room_nights.update_layout(
                title="Room Nights Giornalieri Previsti 2026",
                xaxis_title="Data",
                yaxis_title="Room Nights",
                hovermode='x unified',
                height=350,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_room_nights, use_container_width=True)
        
        with col2:
            fig_revenue = go.Figure()
            
            fig_revenue.add_trace(go.Scatter(
                x=df_forecast['Data'],
                y=df_forecast['Revenue_Forecast'],
                mode='lines',
                name='Revenue',
                line=dict(color='#2ecc71', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ))
            
            fig_revenue.update_layout(
                title="Revenue Giornaliero Previsto 2026",
                xaxis_title="Data",
                yaxis_title="Revenue (€)",
                hovermode='x unified',
                height=350,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Grafico Occupazione
        st.markdown("### 📈 Occupazione Prevista 2026")
        
        fig_occupancy = go.Figure()
        
        fig_occupancy.add_trace(go.Scatter(
            x=df_forecast['Data'],
            y=df_forecast['Occupazione_Forecast'] * 100,
            mode='lines',
            name='Occupazione %',
            line=dict(color='#9b59b6', width=2),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        ))
        
        # Linea target occupazione
        fig_occupancy.add_hline(
            y=occupancy_base * 100,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Target {occupancy_base*100:.0f}%",
            annotation_position="right"
        )
        
        fig_occupancy.update_layout(
            title="Occupazione % Giornaliera con Stagionalità",
            xaxis_title="Data",
            yaxis_title="Occupazione %",
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_occupancy, use_container_width=True)
        
        st.markdown("---")
        
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
            'Room_Nights_Forecast': 'sum',
            'Revenue_Forecast': 'sum',
            'Occupazione_Forecast': 'mean',
            'Data': 'count'
        }).reset_index()
        
        monthly_forecast.columns = ['Mese_Num', 'Mese', 'ADR_Medio', 'ADR_Min', 'ADR_Max', 'ADR_StdDev', 
                                    'Room_Nights', 'Revenue', 'Occupazione', 'Giorni']
        
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
        monthly_forecast['Room_Nights'] = monthly_forecast['Room_Nights'].round(0)
        monthly_forecast['Revenue'] = monthly_forecast['Revenue'].round(0)
        monthly_forecast['Occupazione'] = (monthly_forecast['Occupazione'] * 100).round(1)
        
        # Tabella con tutte le informazioni
        display_df = monthly_forecast[['Mese', 'Giorni', 'ADR_Medio', 'Room_Nights', 'Occupazione', 'Revenue',
                                       'ADR_2025', 'Var_vs_2025_%']].copy()
        
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
            'Room_Nights': '{:,.0f}',
            'Occupazione': '{:.1f}%',
            'Revenue': '€{:,.0f}',
            'ADR_2025': '€{:.2f}',
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
    # TAB 5: BOOKING CURVE & RM
    # =============================
    with tab5:
        st.header("📊 Booking Curve Analysis & Revenue Management")
        
        # Le snapshot sono già caricate (2025 da GitHub + opzionale user, 2026 da sidebar)
        if has_exact_comparable:
            st.success(f"✅ Confronto perfetto attivato: Snapshot 2025 del {exact_comparable_date.strftime('%d/%m/%Y')} caricata!")
        else:
            st.info(f"ℹ️ Confronto con snapshot 2025 più vicina disponibile")
        
        st.markdown("---")
        
        # Calcola confronto - usa data specifica se snapshot comparabile è caricato
        force_comparison_date = exact_comparable_date if has_exact_comparable else None
        
        # DEBUG - Expander per info tecniche
        with st.expander("🔍 Debug Info", expanded=True):
            if force_comparison_date:
                st.info(f"**Forzo confronto con data:** {force_comparison_date.strftime('%d/%m/%Y')}")
            else:
                st.info("**Nessuna data forzata**, userò la più vicina disponibile")
            
            # Mostra chiamata alla funzione con debug attivo
            comparison, df_2025_comparable = compare_booking_curves(
                df_snapshots_2025, 
                df_snapshot_2026,
                force_date=force_comparison_date,
                debug=True  # Attiva debug
            )
        
        # Mostra date di confronto
        booking_window_diff = abs((comparison['snapshot_date_2026'] - comparison['snapshot_date_2025'].replace(year=2025)).days)
        
        st.markdown(f"""
        **Confronto in corso:**
        - 📅 Snapshot 2026: {comparison['snapshot_date_2026'].strftime('%d/%m/%Y')}
        - 📅 Snapshot 2025: {comparison['snapshot_date_2025'].strftime('%d/%m/%Y')}
        - 📊 Differenza booking window: **{booking_window_diff} giorni** {'✅ PERFETTO!' if booking_window_diff == 0 else '⚠️'}
        """)
        
        # Metriche principali
        st.markdown("### 📊 Gap Analysis: 2026 vs 2025")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Room Nights 2026",
                f"{comparison['room_nights_2026']:,.0f}",
                delta=f"{comparison['gap_room_nights']:+,.0f} ({comparison['gap_room_nights_pct']:+.1f}%)",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "ADR Medio 2026",
                f"€{comparison['adr_2026']:.2f}",
                delta=f"€{comparison['gap_adr']:+.2f} ({comparison['gap_adr_pct']:+.1f}%)",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Revenue 2026",
                f"€{comparison['revenue_2026']:,.0f}",
                delta=f"€{comparison['gap_revenue']:+,.0f} ({comparison['gap_revenue_pct']:+.1f}%)",
                delta_color="normal"
            )
        
        with col4:
            snapshot_label = comparison['snapshot_date_2026'].strftime('%d/%m/%Y')
            comparable_label = comparison['snapshot_date_2025'].strftime('%d/%m/%Y')
            st.metric(
                "Data Confronto",
                snapshot_label,
                delta=f"vs {comparable_label}",
                delta_color="off"
            )
        
        st.markdown("---")
        
        # Booking Curve Graph
        st.markdown("### 📈 Booking Curve: Evoluzione 2025 vs OTB 2026")
        
        fig_booking_curve = go.Figure()
        
        # Plot tutte le snapshot 2025
        for snapshot_date in sorted(df_snapshots_2025['Snapshot_Date'].unique()):
            df_snap = df_snapshots_2025[df_snapshots_2025['Snapshot_Date'] == snapshot_date]
            
            # Aggrega per mese
            df_snap_monthly = df_snap.copy()
            df_snap_monthly['Mese'] = df_snap_monthly['Data'].dt.to_period('M')
            monthly_data = df_snap_monthly.groupby('Mese')['Room nights'].sum().reset_index()
            monthly_data['Mese_Str'] = monthly_data['Mese'].dt.strftime('%b %Y')
            
            label = snapshot_date.strftime('%b %Y')
            is_comparable = (snapshot_date == comparison['snapshot_date_2025'])
            
            fig_booking_curve.add_trace(go.Scatter(
                x=monthly_data['Mese_Str'],
                y=monthly_data['Room nights'],
                mode='lines+markers',
                name=f'2025 @ {label}',
                line=dict(width=3 if is_comparable else 1.5, color='red' if is_comparable else 'lightgray'),
                marker=dict(size=8 if is_comparable else 4),
                opacity=1.0 if is_comparable else 0.3
            ))
        
        # Plot snapshot 2026
        df_2026_monthly = df_snapshot_2026.copy()
        df_2026_monthly['Mese'] = df_2026_monthly['Data'].dt.to_period('M')
        monthly_2026 = df_2026_monthly.groupby('Mese')['Room nights'].sum().reset_index()
        monthly_2026['Mese_Str'] = monthly_2026['Mese'].dt.strftime('%b %Y')
        
        fig_booking_curve.add_trace(go.Scatter(
            x=monthly_2026['Mese_Str'],
            y=monthly_2026['Room nights'],
            mode='lines+markers',
            name=f'2026 @ {snapshot_label}',
            line=dict(width=4, color='#2ecc71', dash='solid'),
            marker=dict(size=10, symbol='star')
        ))
        
        fig_booking_curve.update_layout(
            title="Booking Curve Comparison: Come si è riempito il 2025 vs Come si sta riempiendo il 2026",
            xaxis_title="Mese",
            yaxis_title="Room Nights Cumulative",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        st.plotly_chart(fig_booking_curve, use_container_width=True)
        
        # Gap mensile
        st.markdown("### 📅 Gap Analysis per Mese")
        
        # Calcola gap per mese
        df_2025_monthly = df_2025_comparable.copy()
        df_2025_monthly['Mese'] = df_2025_monthly['Data'].dt.month
        df_2025_monthly['Mese_Nome'] = df_2025_monthly['Data'].dt.strftime('%B')
        monthly_2025 = df_2025_monthly.groupby(['Mese', 'Mese_Nome']).agg({
            'Room nights': 'sum',
            'ADR Bed': 'mean'
        }).reset_index()
        
        df_2026_monthly_full = df_snapshot_2026.copy()
        df_2026_monthly_full['Mese'] = df_2026_monthly_full['Data'].dt.month
        df_2026_monthly_full['Mese_Nome'] = df_2026_monthly_full['Data'].dt.strftime('%B')
        monthly_2026_full = df_2026_monthly_full.groupby(['Mese', 'Mese_Nome']).agg({
            'Room nights': 'sum',
            'ADR Bed': 'mean'
        }).reset_index()
        
        # Merge
        monthly_comparison = monthly_2025.merge(
            monthly_2026_full,
            on=['Mese', 'Mese_Nome'],
            how='outer',
            suffixes=('_2025', '_2026')
        ).fillna(0)
        
        monthly_comparison['gap_rn'] = monthly_comparison['Room nights_2026'] - monthly_comparison['Room nights_2025']
        monthly_comparison['gap_pct'] = (monthly_comparison['gap_rn'] / monthly_comparison['Room nights_2025'] * 100).replace([np.inf, -np.inf], 0)
        monthly_comparison['gap_adr'] = monthly_comparison['ADR Bed_2026'] - monthly_comparison['ADR Bed_2025']
        
        # Grafico gap
        fig_gap = go.Figure()
        
        colors_gap = ['#e74c3c' if x >= 0 else '#3498db' for x in monthly_comparison['gap_pct']]
        
        fig_gap.add_trace(go.Bar(
            x=monthly_comparison['Mese_Nome'],
            y=monthly_comparison['gap_pct'],
            text=monthly_comparison['gap_pct'].apply(lambda x: f"{x:+.1f}%"),
            textposition='outside',
            marker_color=colors_gap,
            name='Gap %',
            hovertemplate='%{x}<br>Gap: %{y:.1f}%<extra></extra>'
        ))
        
        fig_gap.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig_gap.add_hline(y=-20, line_dash="dash", line_color="red", line_width=1,
                         annotation_text="Soglia Critica (-20%)", annotation_position="left")
        
        fig_gap.update_layout(
            title="Gap Room Nights per Mese: 2026 vs 2025 (%)",
            xaxis_title="Mese",
            yaxis_title="Gap %",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_gap, use_container_width=True)
        
        # Tabella dettaglio mensile
        display_monthly = monthly_comparison[['Mese_Nome', 'Room nights_2025', 'Room nights_2026', 
                                              'gap_rn', 'gap_pct', 'ADR Bed_2025', 'ADR Bed_2026']].copy()
        display_monthly.columns = ['Mese', 'RN 2025', 'RN 2026', 'Gap RN', 'Gap %', 'ADR 2025', 'ADR 2026']
        
        st.dataframe(
            display_monthly.style.format({
                'RN 2025': '{:,.0f}',
                'RN 2026': '{:,.0f}',
                'Gap RN': '{:+,.0f}',
                'Gap %': '{:+.1f}%',
                'ADR 2025': '€{:.2f}',
                'ADR 2026': '€{:.2f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Pickup forecast
        st.markdown("### 🚀 Pickup Rate & Forecast")
        
        pickup_df = calculate_pickup_rates(df_snapshots_2025)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Grafico pickup
            fig_pickup = go.Figure()
            
            fig_pickup.add_trace(go.Bar(
                x=pickup_df['from_label'],
                y=pickup_df['pickup_total'],
                text=pickup_df['pickup_total'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                marker_color='#3498db',
                name='Pickup Totale'
            ))
            
            fig_pickup.update_layout(
                title="Pickup Room Nights tra Snapshot Consecutive (2025)",
                xaxis_title="Periodo",
                yaxis_title="Room Nights Aggiunte",
                height=350,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_pickup, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Statistiche Pickup")
            
            avg_pickup = pickup_df['pickup_total'].mean()
            max_pickup = pickup_df['pickup_total'].max()
            total_pickup = pickup_df['pickup_total'].sum()
            
            st.metric("Pickup Medio", f"{avg_pickup:,.0f} RN")
            st.metric("Pickup Massimo", f"{max_pickup:,.0f} RN")
            st.metric("Pickup Totale 2025", f"{total_pickup:,.0f} RN")
            
            # Forecast semplice
            rn_attuale_2026 = comparison['room_nights_2026']
            snapshots_rimanenti = 12 - 3  # Assumendo siamo a snapshot #3
            pickup_previsto = avg_pickup * snapshots_rimanenti
            forecast_finale = rn_attuale_2026 + pickup_previsto
            
            st.markdown("---")
            st.markdown("#### 🎯 Forecast Semplice")
            st.metric("RN Attuali 2026", f"{rn_attuale_2026:,.0f}")
            st.metric("Pickup Previsto", f"+{pickup_previsto:,.0f}")
            st.metric("Forecast Finale", f"{forecast_finale:,.0f}")
        
        st.markdown("---")
        
        # Revenue Management Suggestions
        st.markdown("### 💡 Suggerimenti Revenue Management")
        
        suggestions = generate_rm_suggestions(comparison, monthly_comparison, pickup_df)
        
        for suggestion in suggestions:
            if suggestion['type'] == 'critical':
                st.error(f"{suggestion['icon']} **{suggestion['title']}**\n\n{suggestion['message']}")
            elif suggestion['type'] == 'warning':
                st.warning(f"{suggestion['icon']} **{suggestion['title']}**\n\n{suggestion['message']}")
            elif suggestion['type'] == 'success':
                st.success(f"{suggestion['icon']} **{suggestion['title']}**\n\n{suggestion['message']}")
            else:
                st.info(f"{suggestion['icon']} **{suggestion['title']}**\n\n{suggestion['message']}")
            
            if 'actions' in suggestion and len(suggestion['actions']) > 0:
                st.markdown("**Azioni Consigliate:**")
                for action in suggestion['actions']:
                    st.markdown(f"• {action}")
            
            st.markdown("")
    
    # =============================
    # TAB 6: METRICHE DETTAGLIATE
    # =============================
    with tab6:
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
