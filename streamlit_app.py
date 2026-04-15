import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

# Try importing SARIMA
SARIMA_AVAILABLE = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings('ignore')
    SARIMA_AVAILABLE = True
except ImportError:
    pass  # SARIMA not available, will use RF only
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
        
        # Determina quale formato di file è
        if 'Data' in df.columns and df['Data'].dtype != 'object':
            # FORMATO 1: Colonna 'Data' già in datetime
            df['Data'] = pd.to_datetime(df['Data'])
            
        elif 'Data' in df.columns and df['Data'].dtype == 'object':
            # FORMATO 1b: Colonna 'Data' come stringa da parsare
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            
        elif 'Giorno' in df.columns:
            # FORMATO 2: Colonna 'Giorno' in formato "Dom 28/05/2025"
            def parse_date(date_str):
                try:
                    date_only = date_str.split(' ', 1)[-1] if isinstance(date_str, str) else date_str
                    return pd.to_datetime(date_only, format='%d/%m/%Y', errors='coerce')
                except:
                    return pd.NaT
            
            df['Data'] = df['Giorno'].apply(parse_date)
        else:
            st.error("❌ File non riconosciuto: manca colonna 'Data' o 'Giorno'")
            return None
        
        # Filtra righe valide
        df = df.dropna(subset=['Data'])
        df = df[df['ADR Bed'] > 0].copy()
        
        # IMPORTANTE: Gestisci ADR Cam se disponibile
        has_adr_cam = 'ADR Cam' in df.columns and df['ADR Cam'].notna().any()
        
        # IMPORTANTE: Se esiste colonna Room Revenue, usala invece di calcolarla
        if 'Room Revenue' in df.columns:
            # Usa revenue reale dal file
            df['Revenue'] = df['Room Revenue']
            revenue_source = "Room Revenue (file)"
        elif has_adr_cam:
            # Calcola da Room nights × ADR Cam
            df['Revenue'] = df['Room nights'] * df['ADR Cam']
            revenue_source = "RN × ADR Cam"
        else:
            # Fallback: calcola da Room nights × ADR Bed
            df['Revenue'] = df['Room nights'] * df['ADR Bed']
            revenue_source = "RN × ADR Bed"
        
        if len(df) > 0:
            # Usa la data minima come snapshot date (inizio stagione)
            df['Snapshot_Date'] = pd.to_datetime('today')
            df['Snapshot_Label'] = pd.to_datetime('today').strftime('%b %Y')
            
            total_revenue = df['Revenue'].sum()
            
            st.sidebar.write(f"✅ **Snapshot caricata:**")
            st.sidebar.write(f"   • Giorni: {len(df)}")
            st.sidebar.write(f"   • Room Nights: {df['Room nights'].sum():.0f}")
            
            # Mostra ADR Bed e ADR Cam se disponibile
            st.sidebar.write(f"   • ADR Bed: €{df['ADR Bed'].mean():.2f}")
            if has_adr_cam:
                st.sidebar.write(f"   • ADR Cam: €{df['ADR Cam'].mean():.2f}")
                pax_per_room = df['ADR Cam'].mean() / df['ADR Bed'].mean()
                st.sidebar.write(f"   • Pax/Camera: {pax_per_room:.2f}")
            
            st.sidebar.write(f"   • Revenue: €{total_revenue:,.0f}")
            st.sidebar.caption(f"   ({revenue_source})")
            
            return df
        else:
            return None
    
    except Exception as e:
        st.error(f"Errore nel caricamento dello snapshot 2026: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
    # Usa colonna Revenue se disponibile, altrimenti calcola
    if 'Revenue' in df_2025_comparable.columns:
        revenue_2025 = df_2025_comparable['Revenue'].sum()
    else:
        revenue_2025 = (df_2025_comparable['Room nights'] * df_2025_comparable['ADR Bed']).sum()
    
    if 'Revenue' in df_snapshot_2026.columns:
        revenue_2026 = df_snapshot_2026['Revenue'].sum()
    else:
        revenue_2026 = (df_snapshot_2026['Room nights'] * df_snapshot_2026['ADR Bed']).sum()
    
    # Check se ADR Cam è disponibile
    has_adr_cam_2025 = 'ADR Cam' in df_2025_comparable.columns and df_2025_comparable['ADR Cam'].notna().any()
    has_adr_cam_2026 = 'ADR Cam' in df_snapshot_2026.columns and df_snapshot_2026['ADR Cam'].notna().any()
    
    comparison = {
        'snapshot_date_2025': closest_snapshot,
        'snapshot_date_2026': snapshot_date_2026,
        'room_nights_2025': df_2025_comparable['Room nights'].sum(),
        'room_nights_2026': df_snapshot_2026['Room nights'].sum(),
        'adr_bed_2025': df_2025_comparable['ADR Bed'].mean(),
        'adr_bed_2026': df_snapshot_2026['ADR Bed'].mean(),
        'adr_cam_2025': df_2025_comparable['ADR Cam'].mean() if has_adr_cam_2025 else None,
        'adr_cam_2026': df_snapshot_2026['ADR Cam'].mean() if has_adr_cam_2026 else None,
        'revenue_2025': revenue_2025,
        'revenue_2026': revenue_2026,
        # Mantieni retrocompatibilità con 'adr' senza suffisso
        'adr_2025': df_2025_comparable['ADR Bed'].mean(),
        'adr_2026': df_snapshot_2026['ADR Bed'].mean(),
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
            mesi_critici = ', '.join(weak_months['Mese_Nome'].tolist())
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

# Stile CSS personalizzato ispirato al mockup RMS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    /* RMS Style Variables */
    :root {
        --rms-accent: #f0883e;
        --rms-accent2: #58a6ff;
        --rms-green: #3fb950;
        --rms-red: #f85149;
        --rms-yellow: #d29922;
        --rms-surface: #161b22;
        --rms-border: #30363d;
    }
    
    /* Main Headers */
    .main-header {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .kpi-card.success {
        background: linear-gradient(135deg, #3fb950 0%, #2e8b40 100%);
    }
    
    .kpi-card.warning {
        background: linear-gradient(135deg, #d29922 0%, #b8860b 100%);
    }
    
    .kpi-card.critical {
        background: linear-gradient(135deg, #f85149 0%, #d32f2f 100%);
    }
    
    .kpi-card.info {
        background: linear-gradient(135deg, #58a6ff 0%, #0969da 100%);
    }
    
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-delta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Alert Boxes */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin-bottom: 1rem;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .alert-box.critical {
        background: #fee;
        border-color: #f85149;
        color: #721c24;
    }
    
    .alert-box.warning {
        background: #fff3cd;
        border-color: #d29922;
        color: #856404;
    }
    
    .alert-box.success {
        background: #d4edda;
        border-color: #3fb950;
        color: #155724;
    }
    
    .alert-box.info {
        background: #d1ecf1;
        border-color: #58a6ff;
        color: #0c5460;
    }
    
    .alert-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .alert-message {
        margin-bottom: 0.5rem;
    }
    
    .alert-action {
        font-weight: 500;
        font-style: italic;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    /* Tables */
    .dataframe {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.9rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_altri_segmenti():
    """Carica i dati degli altri segmenti (non Diretti) da GitHub"""
    
    try:
        df_2023 = pd.read_excel('altrisegmenti/altrisegmenti_2023.xlsx')
        df_2024 = pd.read_excel('altrisegmenti/altrisegmenti_2024.xlsx')
        df_2025 = pd.read_excel('altrisegmenti/altrisegmenti_2025.xlsx')
        
        return {
            2023: df_2023,
            2024: df_2024,
            2025: df_2025
        }
    except Exception as e:
        st.warning(f"⚠️ File altri segmenti non trovati su GitHub: {str(e)}")
        return None

@st.cache_data
def load_historical_data(file_2023, file_2024, file_2025):
    """Carica e processa i dati storici dalle tre stagioni (SEGMENTI DIRETTI)"""
    
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

def calculate_segment_weight(df_historical, df_altri_segmenti):
    """
    Calcola automaticamente il peso dei segmenti diretti sul totale hotel
    basandosi sui dati storici 2023-2025
    
    Returns:
        peso_diretti: percentuale media (0.0-1.0) dei segmenti diretti
        breakdown: dizionario con dettagli per anno
    """
    
    if df_altri_segmenti is None:
        # Fallback: usa peso empirico medio
        st.sidebar.warning("⚠️ Dati altri segmenti non disponibili, uso peso stimato 22.5%")
        return 0.225, None
    
    breakdown = {}
    
    for year in [2023, 2024, 2025]:
        # Room Nights DIRETTI (SITO WEB, OTA, DIRETTI INDIVIDUALI)
        df_diretti_year = df_historical[df_historical['Anno'] == year]
        rn_diretti = df_diretti_year['Room nights'].sum() if 'Room nights' in df_diretti_year.columns else 0
        
        # Room Nights ALTRI SEGMENTI (GRUPPI, MICE, ESTERO, etc.)
        df_altri_year = df_altri_segmenti[year]
        rn_altri = df_altri_year['Room nights'].sum() if 'Room nights' in df_altri_year.columns else 0
        
        # Totale hotel
        rn_totale = rn_diretti + rn_altri
        peso_year = (rn_diretti / rn_totale) if rn_totale > 0 else 0
        
        breakdown[year] = {
            'rn_diretti': rn_diretti,
            'rn_altri': rn_altri,
            'rn_totale': rn_totale,
            'peso_diretti': peso_year
        }
    
    # Calcola peso medio triennale
    peso_medio = sum([breakdown[y]['peso_diretti'] for y in [2023, 2024, 2025]]) / 3
    
    return peso_medio, breakdown

def calculate_model_metrics(models, df_historical):
    """Calcola metriche di accuracy per i modelli"""
    
    required_cols = ['Giorno_Stagione', 'Settimana_Anno', 'Giorno_Settimana', 'Mese', 'ADR Bed']
    df_train = df_historical[required_cols].dropna()
    
    X = df_train[['Giorno_Stagione', 'Settimana_Anno', 'Giorno_Settimana', 'Mese']].values
    y_true = df_train['ADR Bed'].values
    
    metrics = {}
    
    # Random Forest
    if 'RandomForest' in models:
        rf_model = models['RandomForest']
        y_pred = rf_model.predict(X)
        
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = rf_model.score(X, y_true)
        
        metrics['RandomForest'] = {
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'feature_importance': dict(zip(
                ['Giorno_Stagione', 'Settimana_Anno', 'Giorno_Settimana', 'Mese'],
                rf_model.feature_importances_
            ))
        }
    
    return metrics

def generate_automatic_alerts(comparison, monthly_comparison, df_forecast):
    """Genera alert automatici basati su analisi dati"""
    
    alerts = []
    
    # Alert 1: Gap Room Nights totale
    gap_pct = comparison.get('gap_room_nights_pct', 0)
    if gap_pct < -15:
        alerts.append({
            'severity': 'critical',
            'icon': '🔴',
            'title': 'CRITICO: Booking Pace Molto Lento',
            'message': f"Sei {abs(gap_pct):.1f}% indietro rispetto al 2025 ({abs(comparison.get('gap_room_nights', 0)):.0f} RN)",
            'action': 'Attiva immediatamente campagne promozionali e rivedi pricing'
        })
    elif gap_pct < -5:
        alerts.append({
            'severity': 'warning',
            'icon': '🟡',
            'title': 'ATTENZIONE: Booking Pace Lento',
            'message': f"Sei {abs(gap_pct):.1f}% indietro rispetto al 2025 ({abs(comparison.get('gap_room_nights', 0)):.0f} RN)",
            'action': 'Monitora quotidianamente e prepara azioni correttive'
        })
    elif gap_pct > 10:
        alerts.append({
            'severity': 'success',
            'icon': '🟢',
            'title': 'OTTIMO: Booking Pace Forte!',
            'message': f"Sei {gap_pct:.1f}% avanti rispetto al 2025 (+{comparison.get('gap_room_nights', 0):.0f} RN)",
            'action': 'Considera aumento tariffe per massimizzare revenue'
        })
    
    # Alert 2: Mesi critici
    if monthly_comparison is not None and len(monthly_comparison) > 0:
        # Verifica che monthly_comparison abbia le colonne necessarie
        if 'gap_pct' in monthly_comparison.columns:
            weak_months = monthly_comparison[monthly_comparison['gap_pct'] < -20]
            if len(weak_months) > 0:
                if 'Mese_Nome' in weak_months.columns:
                    mesi_critici = ', '.join(weak_months['Mese_Nome'].tolist())
                else:
                    mesi_critici = ', '.join([str(m) for m in weak_months['Mese'].tolist()])
                
                alerts.append({
                    'severity': 'critical',
                    'icon': '📅',
                    'title': f'Mesi Critici: {mesi_critici}',
                    'message': f'{len(weak_months)} mesi con gap > -20%',
                    'action': f'Focus immediato su {mesi_critici} con promo dedicate'
                })
    
    # Alert 3: ADR trend
    adr_gap_pct = comparison.get('gap_adr_pct', 0)
    if adr_gap_pct < -5:
        alerts.append({
            'severity': 'warning',
            'icon': '💰',
            'title': 'ADR in Calo',
            'message': f"ADR {adr_gap_pct:.1f}% vs 2025 (€{comparison.get('gap_adr', 0):.2f})",
            'action': 'Rivedi strategia pricing e chiudi canali discount'
        })
    
    # Alert 4: Day-type analysis
    if df_forecast is not None and len(df_forecast) > 0:
        df_forecast['Giorno_Nome_IT'] = df_forecast['Giorno_Settimana'].map({
            0: 'Lunedì', 1: 'Martedì', 2: 'Mercoledì', 3: 'Giovedì',
            4: 'Venerdì', 5: 'Sabato', 6: 'Domenica'
        })
        
        dow_avg = df_forecast.groupby('Giorno_Nome_IT')['ADR_Bed_Forecast'].mean()
        
        # Trova giorni deboli
        weak_days = dow_avg[dow_avg < dow_avg.mean() * 0.85]
        if len(weak_days) > 0:
            giorni_deboli = ', '.join(weak_days.index.tolist())
            alerts.append({
                'severity': 'info',
                'icon': '📊',
                'title': 'Pattern Giorni Deboli',
                'message': f'{giorni_deboli} costantemente sotto media',
                'action': 'Considera promo specifiche per questi giorni'
            })
    
    return alerts

def analyze_day_type_performance(df_forecast, df_historical):
    """Analizza performance per tipo di giorno"""
    
    df_forecast['Giorno_Nome_IT'] = df_forecast['Giorno_Settimana'].map({
        0: 'Lunedì', 1: 'Martedì', 2: 'Mercoledì', 3: 'Giovedì',
        4: 'Venerdì', 5: 'Sabato', 6: 'Domenica'
    })
    
    # Performance forecast 2026
    forecast_perf = df_forecast.groupby('Giorno_Nome_IT').agg({
        'ADR_Bed_Forecast': 'mean',
        'Occupazione_Forecast': 'mean'
    }).reset_index()
    
    forecast_perf.columns = ['Giorno', 'ADR_2026', 'OCC_2026']
    
    # Performance storica 2025
    df_historical['Giorno_Nome_IT'] = df_historical['Giorno_Settimana'].map({
        0: 'Lunedì', 1: 'Martedì', 2: 'Mercoledì', 3: 'Giovedì',
        4: 'Venerdì', 5: 'Sabato', 6: 'Domenica'
    })
    
    df_2025 = df_historical[df_historical['Anno'] == 2025]
    hist_perf = df_2025.groupby('Giorno_Nome_IT')['ADR Bed'].mean().reset_index()
    hist_perf.columns = ['Giorno', 'ADR_2025']
    
    # Merge
    perf = forecast_perf.merge(hist_perf, on='Giorno', how='left')
    perf['Var_%'] = ((perf['ADR_2026'] - perf['ADR_2025']) / perf['ADR_2025'] * 100).round(1)
    
    # Ordina giorni settimana
    giorni_ordine = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
    perf['Giorno'] = pd.Categorical(perf['Giorno'], categories=giorni_ordine, ordered=True)
    perf = perf.sort_values('Giorno')
    
    return perf

def analyze_weekly_split(df_snapshot_2026, df_snapshots_2025, df_forecast):
    """
    Analizza split settimanale (Dom-Dom) per soggiorni 7 notti
    Confronta: Snapshot 2026 vs Snapshot 2025 stesso periodo vs Chiusura 2025
    """
    
    if df_snapshot_2026 is None:
        return None
    
    # Crea settimane Domenica-Domenica
    df_2026 = df_snapshot_2026.copy()
    df_2026['Settimana_Dom'] = df_2026['Data'].apply(
        lambda x: (x - pd.Timedelta(days=x.dayofweek + 1 if x.dayofweek != 6 else 0)).isocalendar()[1]
    )
    df_2026['Anno_Settimana'] = df_2026['Data'].dt.year.astype(str) + '-W' + df_2026['Settimana_Dom'].astype(str)
    
    # Aggrega per settimana
    weekly_2026 = df_2026.groupby('Anno_Settimana').agg({
        'Room nights': 'sum',
        'ADR Bed': 'mean',
        'Data': ['min', 'max']
    }).reset_index()
    
    weekly_2026.columns = ['Settimana', 'RN_OTB_2026', 'ADR_OTB_2026', 'Data_Inizio', 'Data_Fine']
    
    # Se abbiamo snapshot 2025, confronta
    if df_snapshots_2025 is not None and len(df_snapshots_2025) > 0:
        # Trova snapshot 2025 comparabile (stesso periodo anno scorso)
        snapshot_date_2026 = df_2026['Data'].min()
        comparable_date_2025 = snapshot_date_2026.replace(year=2024)
        
        # Filtra snapshot 2025 più vicina
        df_snapshots_2025['Date_Diff'] = abs((df_snapshots_2025['Snapshot_Date'] - comparable_date_2025).dt.days)
        closest_snapshot = df_snapshots_2025[df_snapshots_2025['Date_Diff'] == df_snapshots_2025['Date_Diff'].min()]
        
        if len(closest_snapshot) > 0:
            df_2025_comp = closest_snapshot.copy()
            df_2025_comp['Settimana_Dom'] = df_2025_comp['Data'].apply(
                lambda x: (x - pd.Timedelta(days=x.dayofweek + 1 if x.dayofweek != 6 else 0)).isocalendar()[1]
            )
            df_2025_comp['Anno_Settimana'] = '2025-W' + df_2025_comp['Settimana_Dom'].astype(str)
            
            weekly_2025_comp = df_2025_comp.groupby('Anno_Settimana').agg({
                'Room nights': 'sum',
                'ADR Bed': 'mean'
            }).reset_index()
            
            weekly_2025_comp.columns = ['Settimana', 'RN_OTB_2025', 'ADR_OTB_2025']
            
            # Merge
            weekly_2026 = weekly_2026.merge(
                weekly_2025_comp[['Settimana', 'RN_OTB_2025', 'ADR_OTB_2025']],
                left_on='Settimana',
                right_on='Settimana',
                how='left',
                suffixes=('', '_2025')
            )
            
            # Calcola gap
            weekly_2026['Gap_RN'] = weekly_2026['RN_OTB_2026'] - weekly_2026['RN_OTB_2025']
            weekly_2026['Gap_RN_Pct'] = (weekly_2026['Gap_RN'] / weekly_2026['RN_OTB_2025'] * 100).round(1)
    
    return weekly_2026

@st.cache_data
def load_budget_2026(uploaded_file):
    """Carica il file budget 2026 mensile"""
    
    try:
        df_budget = pd.read_excel(uploaded_file)
        
        # Mappa mesi italiani
        mese_map = {
            'Gen': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Mag': 5, 'Giu': 6,
            'Lug': 7, 'Ago': 8, 'Set': 9, 'Ott': 10, 'Nov': 11, 'Dic': 12
        }
        
        # Estrai mese da 'Mese Anno'
        if 'Mese Anno' in df_budget.columns:
            df_budget['Mese_Estratto'] = df_budget['Mese Anno'].apply(
                lambda x: mese_map.get(str(x).split()[0], 0) if pd.notna(x) else 0
            )
        
        # Filtra solo righe con dati budget validi
        df_budget = df_budget[df_budget['Room Revenue BDG'].notna()].copy()
        
        return df_budget
    
    except Exception as e:
        st.error(f"Errore caricamento budget: {str(e)}")
        return None

def calculate_daily_gap_vs_budget(df_forecast, df_budget, df_snapshot_2026, df_historical):
    """
    Calcola gap giornaliero vs budget usando distribuzione INTELLIGENTE
    basata su pattern storici e trend, non distribuzione uniforme
    """
    
    if df_budget is None or len(df_budget) == 0:
        return None
    
    # Aggiungi mese al forecast
    df_forecast['Mese'] = df_forecast['Data'].dt.month
    df_forecast['Giorno_Mese'] = df_forecast['Data'].dt.day
    
    # Per ogni mese con budget, calcola distribuzione intelligente basata su storico
    for idx, budget_row in df_budget.iterrows():
        mese = budget_row['Mese_Estratto']
        
        if pd.isna(mese) or mese == 0:
            continue
        
        # Budget mensile totale
        adr_budget_mese = budget_row['ADR Bed BDG']
        rn_budget_mese = budget_row['Roomnights BDG']
        revenue_budget_mese = budget_row['Room Revenue BDG']
        
        if pd.isna(adr_budget_mese) or pd.isna(rn_budget_mese):
            continue
        
        # ALGORITMO INTELLIGENTE: Analizza distribuzione storica dello stesso mese
        # Prendi dati storici dello stesso mese (2023, 2024, 2025)
        df_mese_storico = df_historical[df_historical['Mese'] == mese].copy()
        
        if len(df_mese_storico) > 0:
            # Calcola "peso" di ogni giorno del mese basato su storico
            # Peso = ADR storico medio di quel giorno settimana in quel mese
            
            # Group by giorno settimana per trovare pattern
            dow_pattern = df_mese_storico.groupby('Giorno_Settimana').agg({
                'ADR Bed': 'mean',
                'Room nights': 'mean'
            }).reset_index()
            
            dow_pattern.columns = ['Giorno_Settimana', 'ADR_Storico_Medio', 'RN_Storico_Medio']
            
            # Normalizza i pesi (somma = 1)
            total_adr_weight = dow_pattern['ADR_Storico_Medio'].sum()
            dow_pattern['ADR_Weight'] = dow_pattern['ADR_Storico_Medio'] / total_adr_weight if total_adr_weight > 0 else 1/7
            
            # Merge pattern con forecast del mese
            df_mese_forecast = df_forecast[df_forecast['Mese'] == mese].copy()
            df_mese_forecast = df_mese_forecast.merge(
                dow_pattern[['Giorno_Settimana', 'ADR_Storico_Medio', 'ADR_Weight']],
                on='Giorno_Settimana',
                how='left'
            )
            
            # DISTRIBUZIONE INTELLIGENTE del budget mensile
            # Invece di Budget_Mese / Giorni, usa peso storico
            giorni_nel_mese = len(df_mese_forecast)
            
            # ADR Budget per giorno = ADR medio mensile × (peso relativo del giorno)
            # Questo significa: weekend avrà ADR budget più alto, weekday più basso
            adr_medio_mese = adr_budget_mese
            
            df_mese_forecast['ADR_Budget_Daily'] = df_mese_forecast.apply(
                lambda row: adr_medio_mese * (1 + (row['ADR_Weight'] - 1/7) * 2) 
                if pd.notna(row.get('ADR_Weight')) else adr_medio_mese,
                axis=1
            )
            
            # RN Budget per giorno = distribuito proporzionalmente ai giorni
            # Ma considerando che alcuni giorni (weekend) attraggono più RN
            df_mese_forecast['RN_Budget_Daily'] = rn_budget_mese / giorni_nel_mese
            
            # Revenue Budget = ADR Budget × RN Budget
            df_mese_forecast['Revenue_Budget_Daily'] = (
                df_mese_forecast['ADR_Budget_Daily'] * df_mese_forecast['RN_Budget_Daily']
            )
            
            # Update nel dataframe principale
            for col in ['ADR_Budget_Daily', 'RN_Budget_Daily', 'Revenue_Budget_Daily']:
                df_forecast.loc[df_mese_forecast.index, col] = df_mese_forecast[col]
        
        else:
            # Fallback: distribuzione uniforme se non ci sono dati storici
            giorni_nel_mese = len(df_forecast[df_forecast['Mese'] == mese])
            
            df_forecast.loc[df_forecast['Mese'] == mese, 'ADR_Budget_Daily'] = adr_budget_mese
            df_forecast.loc[df_forecast['Mese'] == mese, 'RN_Budget_Daily'] = rn_budget_mese / giorni_nel_mese if giorni_nel_mese > 0 else 0
            df_forecast.loc[df_forecast['Mese'] == mese, 'Revenue_Budget_Daily'] = revenue_budget_mese / giorni_nel_mese if giorni_nel_mese > 0 else 0
    
    # Calcola GAP (forecast vs budget intelligente)
    df_forecast['Gap_ADR'] = df_forecast['ADR_Bed_Forecast'] - df_forecast['ADR_Budget_Daily']
    df_forecast['Gap_ADR_Pct'] = (df_forecast['Gap_ADR'] / df_forecast['ADR_Budget_Daily'] * 100).round(1)
    
    df_forecast['Gap_Revenue'] = df_forecast['Revenue_Forecast'] - df_forecast['Revenue_Budget_Daily']
    df_forecast['Gap_Revenue_Pct'] = (df_forecast['Gap_Revenue'] / df_forecast['Revenue_Budget_Daily'] * 100).round(1)
    
    return df_forecast

def generate_pricing_recommendations(df_forecast, df_snapshot_2026, df_historical):
    """
    Genera raccomandazioni pricing INTELLIGENTI basate su:
    1. Gap vs budget (già calcolato con distribuzione smart)
    2. Trend storico dello stesso giorno/periodo
    3. Performance relativa vs anni precedenti
    4. Elasticità implicita (se alzo ADR, cosa succede a RN?)
    """
    
    recommendations = []
    
    # Filtra solo giorni con gap significativo
    df_gap = df_forecast[
        (df_forecast['Gap_ADR'].notna()) & 
        (df_forecast['Gap_ADR'] < -3)  # Almeno €3 sotto budget
    ].copy()
    
    for idx, row in df_gap.iterrows():
        # Dati correnti
        adr_current = row['ADR_Bed_Forecast']
        adr_budget = row['ADR_Budget_Daily']
        gap_adr = row['Gap_ADR']
        
        # Analisi TREND STORICO per questo specifico periodo
        # Trova stesso periodo (±3 giorni) negli anni precedenti
        giorno_stagione = row['Giorno_Stagione']
        
        df_same_period = df_historical[
            (df_historical['Giorno_Stagione'] >= giorno_stagione - 3) &
            (df_historical['Giorno_Stagione'] <= giorno_stagione + 3)
        ].copy()
        
        if len(df_same_period) > 0:
            # ADR storico medio dello stesso periodo
            adr_storico_avg = df_same_period['ADR Bed'].mean()
            
            # Trend: confronta forecast con storico
            trend_vs_storico = ((adr_current - adr_storico_avg) / adr_storico_avg * 100) if adr_storico_avg > 0 else 0
            
            # RACCOMANDAZIONE INTELLIGENTE
            # Se budget > storico → target aggressivo
            # Se budget ≈ storico → target realistico
            # Se budget < storico → segnala problema
            
            if adr_budget > adr_storico_avg * 1.1:
                # Budget ambizioso (+10% vs storico)
                recommendation_type = 'aggressive'
                # Target = 80% del gap (più conservativo)
                adr_recommended = adr_current + (abs(gap_adr) * 0.8)
            elif adr_budget > adr_storico_avg * 0.9:
                # Budget realistico (±10% vs storico)
                recommendation_type = 'realistic'
                # Target = 100% del gap
                adr_recommended = adr_budget
            else:
                # Budget conservativo
                recommendation_type = 'conservative'
                # Target = budget (facile da raggiungere)
                adr_recommended = adr_budget
        else:
            # Nessun dato storico, usa solo budget
            recommendation_type = 'budget_based'
            adr_recommended = adr_budget
            adr_storico_avg = adr_current
            trend_vs_storico = 0
        
        # Revenue recovery potenziale
        rn_forecast = row.get('Room_Nights_Forecast', 0)
        revenue_gain = (adr_recommended - adr_current) * rn_forecast if rn_forecast > 0 else 0
        
        # Priority score SMART
        # Fattori: gap absoluto, weekend, alta stagione, trend storico
        is_weekend = row['Giorno_Settimana'] in [5, 6]
        is_high_season = row['Mese'] in [6, 7, 8]
        
        priority_score = abs(gap_adr)
        if is_weekend:
            priority_score *= 2
        if is_high_season:
            priority_score *= 1.5
        if trend_vs_storico < -10:  # Sotto trend storico
            priority_score *= 1.3
        
        # Severity INTELLIGENTE
        gap_pct = abs(row['Gap_ADR_Pct'])
        
        # Se budget molto > storico, severity più bassa (budget ambizioso)
        if adr_budget > adr_storico_avg * 1.15:
            # Budget ambizioso, allenta severity
            if gap_pct > 30:
                severity = 'critical'
                severity_icon = '🔴'
            elif gap_pct > 15:
                severity = 'warning'
                severity_icon = '🟡'
            else:
                severity = 'info'
                severity_icon = '🔵'
        else:
            # Budget normale, severity standard
            if gap_pct > 20:
                severity = 'critical'
                severity_icon = '🔴'
            elif gap_pct > 10:
                severity = 'warning'
                severity_icon = '🟡'
            else:
                severity = 'info'
                severity_icon = '🔵'
        
        # Costruisci messaggio raccomandazione
        adr_increase_pct = ((adr_recommended - adr_current) / adr_current * 100) if adr_current > 0 else 0
        
        if recommendation_type == 'aggressive':
            action_msg = f"Alza ADR a €{adr_recommended:.2f} (+{adr_increase_pct:.1f}%) - Target parziale (budget ambizioso)"
        elif recommendation_type == 'realistic':
            action_msg = f"Alza ADR a €{adr_recommended:.2f} (+{adr_increase_pct:.1f}%) - Allineamento budget"
        elif recommendation_type == 'conservative':
            action_msg = f"Alza ADR a €{adr_recommended:.2f} (+{adr_increase_pct:.1f}%) - Facile da raggiungere"
        else:
            action_msg = f"Alza ADR a €{adr_recommended:.2f} (+{adr_increase_pct:.1f}%)"
        
        # Giorno nome
        giorno_nome = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom'][row['Giorno_Settimana']]
        
        recommendations.append({
            'Data': row['Data'],
            'Giorno_Nome': giorno_nome,
            'ADR_Current': adr_current,
            'ADR_Budget': adr_budget,
            'ADR_Storico': adr_storico_avg,
            'ADR_Recommended': adr_recommended,
            'Gap_EUR': gap_adr,
            'Gap_Pct': row['Gap_ADR_Pct'],
            'Trend_vs_Storico': trend_vs_storico,
            'Revenue_Gain': revenue_gain,
            'Priority_Score': priority_score,
            'Severity': severity,
            'Severity_Icon': severity_icon,
            'Recommendation_Type': recommendation_type,
            'Is_Weekend': is_weekend,
            'Is_High_Season': is_high_season,
            'Action': action_msg
        })
    
    # Converti in DataFrame e ordina per priority
    if len(recommendations) > 0:
        df_recommendations = pd.DataFrame(recommendations)
        df_recommendations = df_recommendations.sort_values('Priority_Score', ascending=False)
        return df_recommendations
    
    return None

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
    
    # 4. SARIMA (se disponibile) - ottimo per dati stagionali
    if SARIMA_AVAILABLE:
        st.sidebar.info("✅ SARIMA disponibile - training in corso...")
        try:
            # Prepara serie temporale
            df_ts = df_train.copy()
            df_ts = df_ts.sort_values('Giorno_Stagione')
            df_ts = df_ts.set_index(pd.date_range(start='2023-05-01', periods=len(df_ts), freq='D'))
            
            # SARIMA con parametri per stagionalità settimanale
            sarima_model = SARIMAX(df_ts['ADR Bed'], 
                                   order=(1, 1, 1),  # (p, d, q)
                                   seasonal_order=(1, 1, 1, 7),  # (P, D, Q, s) - stagionalità settimanale
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
            sarima_fit = sarima_model.fit(disp=False, maxiter=50)
            models['SARIMA'] = sarima_fit
            st.sidebar.success("✅ SARIMA allenato con successo!")
        except Exception as e:
            st.sidebar.error(f"❌ SARIMA training fallito: {str(e)[:100]}")
            models['SARIMA'] = None
    else:
        st.sidebar.warning("⚠️ statsmodels non disponibile - usando solo RF")
        models['SARIMA'] = None
    
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
    
    # NUOVO APPROCCIO: Usa principalmente RF (più stabile per hotel stagionale)
    # Poly crea picchi irrealistici, quindi lo usiamo minimamente
    if 'SARIMA' in models and models['SARIMA'] is not None:
        try:
            # Predici con SARIMA
            sarima_pred = models['SARIMA'].forecast(steps=len(df_2026))
            df_2026['ADR_Bed_SARIMA'] = sarima_pred.values
            
            # Ensemble: 80% RF + 20% SARIMA (entrambi stabili)
            df_2026['ADR_Bed_Ensemble'] = (
                0.80 * pred_rf +
                0.20 * df_2026['ADR_Bed_SARIMA']
            )
        except Exception as e:
            # Fallback: usa solo RF
            df_2026['ADR_Bed_Ensemble'] = pred_rf
    else:
        # Se SARIMA non disponibile, usa solo RF (più stabile di Poly)
        df_2026['ADR_Bed_Ensemble'] = pred_rf
    
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
        
        # NUOVO: Carica dati altri segmenti
        with st.spinner('Caricamento dati altri segmenti...'):
            df_altri_segmenti = load_altri_segmenti()
        
        # Calcola peso segmenti diretti automaticamente
        peso_diretti, breakdown_segmenti = calculate_segment_weight(df_historical, df_altri_segmenti)
        
        # Mostra info di debug
        with st.sidebar.expander("🔍 Info Dataset", expanded=False):
            st.write(f"**Anni presenti:** {sorted(df_historical['Anno'].unique())}")
            st.write(f"**Periodo:** {df_historical['Data'].min().strftime('%d/%m/%Y')} - {df_historical['Data'].max().strftime('%d/%m/%Y')}")
            st.write(f"**ADR medio:** €{df_historical['ADR Bed'].mean():.2f}")
            st.write(f"**Colonne disponibili:** {len(df_historical.columns)}")
            
            # NUOVO: Info breakdown segmenti
            if breakdown_segmenti:
                st.markdown("---")
                st.markdown("**📊 Breakdown Segmenti:**")
                for year, data in breakdown_segmenti.items():
                    st.write(f"{year}: Diretti {data['peso_diretti']*100:.1f}% ({data['rn_diretti']:,.0f} RN)")
        
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
                
                # Tenta di caricare altri segmenti da GitHub anche se i dati diretti sono manuali
                df_altri_segmenti = load_altri_segmenti()
                peso_diretti, breakdown_segmenti = calculate_segment_weight(df_historical, df_altri_segmenti)
                
                # Mostra info di debug
                with st.sidebar.expander("🔍 Info Dataset", expanded=False):
                    st.write(f"**Anni presenti:** {sorted(df_historical['Anno'].unique())}")
                    st.write(f"**Periodo:** {df_historical['Data'].min().strftime('%d/%m/%Y')} - {df_historical['Data'].max().strftime('%d/%m/%Y')}")
                    st.write(f"**ADR medio:** €{df_historical['ADR Bed'].mean():.2f}")
                    st.write(f"**Colonne disponibili:** {len(df_historical.columns)}")
                    
                    # Info breakdown segmenti
                    if breakdown_segmenti:
                        st.markdown("---")
                        st.markdown("**📊 Breakdown Segmenti:**")
                        for year, data in breakdown_segmenti.items():
                            st.write(f"{year}: Diretti {data['peso_diretti']*100:.1f}% ({data['rn_diretti']:,.0f} RN)")
            
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
    
    # NUOVO: Upload Budget 2026
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Budget 2026")
    
    uploaded_budget_2026 = st.sidebar.file_uploader(
        "**Budget 2026 Mensile** (Opzionale)",
        type=['xlsx', 'xls'],
        key='sidebar_budget_2026',
        help="Budget mensile 2026 per generare raccomandazioni pricing"
    )
    
    # Carica budget se presente
    df_budget_2026 = None
    if uploaded_budget_2026:
        df_budget_2026 = load_budget_2026(uploaded_budget_2026)
        if df_budget_2026 is not None:
            st.sidebar.success(f"✅ Budget caricato: {len(df_budget_2026)} mesi")
    
    # Variabile per snapshot 2025 comparabile (opzionale ma consigliata)
    # NON inizializzare a None qui, altrimenti cancella il file uploader!
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
            
            # Salva in session_state per persistenza
            if uploaded_snapshot_2025_comparable:
                st.session_state['uploaded_snapshot_2025_comparable'] = uploaded_snapshot_2025_comparable
                st.session_state['snapshot_2026_date'] = snapshot_2026_date
            
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
    
    # Configurazione Room Nights con breakdown segmenti
    st.sidebar.markdown("### 🏨 Configurazione Struttura")
    
    camere_totali_hotel = 308  # Fisse
    st.sidebar.metric("Camere Totali Hotel", f"{camere_totali_hotel}")
    
    # Mostra breakdown segmenti
    st.sidebar.markdown("#### 📊 Distribuzione Storica")
    
    if breakdown_segmenti:
        st.sidebar.write(f"**Segmenti Diretti:** {peso_diretti*100:.1f}%")
        st.sidebar.caption("_(SITO WEB, OTA, DIRETTI IND.)_")
        st.sidebar.write(f"**Altri Segmenti:** {(1-peso_diretti)*100:.1f}%")
        st.sidebar.caption("_(GRUPPI, MICE, ESTERO, etc.)_")
        
        # Dettaglio per anno
        with st.sidebar.expander("📈 Dettaglio per Anno", expanded=False):
            for year in [2023, 2024, 2025]:
                data = breakdown_segmenti[year]
                st.write(f"**{year}:**")
                st.write(f"  • Diretti: {data['rn_diretti']:,.0f} RN ({data['peso_diretti']*100:.1f}%)")
                st.write(f"  • Altri: {data['rn_altri']:,.0f} RN")
                st.write(f"  • Totale: {data['rn_totale']:,.0f} RN")
    
    # Calcola camere efficaci per segmenti diretti
    camere_diretti_calcolate = int(camere_totali_hotel * peso_diretti)
    
    st.sidebar.markdown("#### 🎯 Camere Efficaci Segmenti Diretti")
    
    # Opzione override manuale
    use_override = st.sidebar.checkbox(
        "⚙️ Override manuale camere", 
        value=False,
        help="Usa un valore personalizzato invece del calcolo automatico"
    )
    
    if use_override:
        camere_totali = st.sidebar.number_input(
            "Camere Diretti (manuale)",
            min_value=10,
            max_value=308,
            value=camere_diretti_calcolate,
            step=1,
            help="Imposta manualmente le camere per i segmenti diretti"
        )
        st.sidebar.warning("⚠️ Stai usando un valore manuale!")
    else:
        camere_totali = camere_diretti_calcolate
        st.sidebar.metric(
            "Camere Diretti (auto)", 
            f"{camere_totali}",
            help=f"Calcolate automaticamente: {camere_totali_hotel} × {peso_diretti*100:.1f}%"
        )
        st.sidebar.success(f"✅ Calcolate da storico 2023-2025")
    
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
    
    # Tab principale - Nuova struttura ispirata a RMS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Dashboard",
        "📈 Forecast 2026", 
        "🔍 Model Analysis",
        "📉 Booking Curve & RM",
        "📅 Analisi Mensile",
        "🔬 Comparazione Anni",
        "💾 Export & Reports",
        "💰 Pricing Recommendations"
    ])
    
    # Calcola seasonality e modelli
    
    # Variabili per tracking snapshot comparabile (DEVONO essere qui per essere accessibili in tutti i tab)
    has_exact_comparable = False
    exact_comparable_date = None
    df_snapshots_2025 = None
    df_snapshot_2026 = None
    
    # Calcola ratio ADR Cam/Bed dai dati storici (se disponibile)
    adr_cam_bed_ratio = None
    if 'ADR Cam' in df_historical.columns and 'ADR Bed' in df_historical.columns:
        df_valid = df_historical[(df_historical['ADR Cam'].notna()) & (df_historical['ADR Bed'].notna()) & (df_historical['ADR Bed'] > 0)]
        if len(df_valid) > 0:
            adr_cam_bed_ratio = (df_valid['ADR Cam'] / df_valid['ADR Bed']).mean()
            st.sidebar.info(f"📊 Ratio ADR Cam/Bed storico: {adr_cam_bed_ratio:.2f}x")
    
    try:
        with st.spinner('Costruzione modelli predittivi...'):
            seasonality = calculate_seasonality_index(df_historical)
            models = build_forecast_models(df_historical)
            df_forecast_base = generate_forecast_2026(df_historical, models, seasonality, scenario)
        
        # Carica snapshot 2025 e 2026 per forecast ibrido
        with st.spinner('Caricamento snapshot OTB...'):
            df_snapshots_2025 = load_snapshots_2025()
            
            # Recupera snapshot comparabile da session_state se disponibile
            uploaded_snapshot_2025_comparable = st.session_state.get('uploaded_snapshot_2025_comparable', None)
            snapshot_2026_date_from_state = st.session_state.get('snapshot_2026_date', snapshot_2026_date)
            
            # Se disponibile, usa snapshot 2025 comparabile caricato dall'utente
            if uploaded_snapshot_2025_comparable:
                df_snapshot_2025_user = load_snapshot_2026(uploaded_snapshot_2025_comparable)  # Riusa la stessa funzione
                if df_snapshot_2025_user is not None:
                    # Calcola data target (usa quella salvata in session_state)
                    target_date = snapshot_2026_date_from_state.replace(year=2024)
                    
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
            
            # Calcola ADR Cam dal ratio storico (se disponibile)
            if adr_cam_bed_ratio is not None:
                df_forecast['ADR_Cam_Forecast'] = df_forecast['ADR_Bed_Forecast'] * adr_cam_bed_ratio
            else:
                # Fallback: usa pax_per_camera_medio come proxy
                df_forecast['ADR_Cam_Forecast'] = df_forecast['ADR_Bed_Forecast'] * pax_per_camera_medio
            
            # Calcola Revenue usando ADR Cam (più accurato)
            if adr_cam_bed_ratio is not None:
                df_forecast['Revenue_Forecast'] = df_forecast['Room_Nights_Forecast'] * df_forecast['ADR_Cam_Forecast']
            else:
                # Fallback originale
                df_forecast['Revenue_Forecast'] = df_forecast['Bed_Nights_Forecast'] * df_forecast['ADR_Bed_Forecast']
            
            # Calcola RevPAR
            df_forecast['RevPAR_Forecast'] = df_forecast['ADR_Bed_Forecast'] * df_forecast['Occupazione_Forecast']
            
            metrics = calculate_forecast_metrics(df_forecast, df_historical)
            
            # NUOVO: Calcola metriche modelli
            model_metrics = calculate_model_metrics(models, df_historical)
            
            # NUOVO: Analizza performance per tipo di giorno
            day_type_performance = analyze_day_type_performance(df_forecast, df_historical)
            
            # NUOVO: Analizza split settimanale Dom-Dom
            weekly_split_analysis = analyze_weekly_split(df_snapshot_2026, df_snapshots_2025, df_forecast)
            
            # NUOVO: Calcola gap vs budget e raccomandazioni pricing
            df_forecast_with_budget = None
            pricing_recommendations = None
            
            if df_budget_2026 is not None:
                with st.spinner('Calcolo gap vs budget e raccomandazioni pricing intelligenti...'):
                    df_forecast_with_budget = calculate_daily_gap_vs_budget(
                        df_forecast.copy(), 
                        df_budget_2026, 
                        df_snapshot_2026,
                        df_historical  # NUOVO: passa storico per analisi trend
                    )
                    
                    if df_forecast_with_budget is not None:
                        # Usa il forecast con budget invece di quello base
                        df_forecast = df_forecast_with_budget
                        
                        # Genera raccomandazioni INTELLIGENTI
                        pricing_recommendations = generate_pricing_recommendations(
                            df_forecast,
                            df_snapshot_2026,
                            df_historical  # NUOVO: passa storico per trend analysis
                        )
                        
                        if pricing_recommendations is not None:
                            st.sidebar.success(f"✅ {len(pricing_recommendations)} raccomandazioni pricing generate")
        
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
    # TAB 1: DASHBOARD
    # =============================
    with tab1:
        st.header("🎯 Dashboard Revenue Management")
        
        # Calcola confronto per dashboard
        if df_snapshots_2025 is not None and df_snapshot_2026 is not None:
            force_comparison_date = exact_comparable_date if has_exact_comparable else None
            comparison_dashboard, _ = compare_booking_curves(
                df_snapshots_2025, 
                df_snapshot_2026,
                force_date=force_comparison_date,
                debug=False
            )
            
            # KPI Cards Row 1
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                var_color = 'success' if metrics.get('Variazione_vs_2025', 0) > 0 else 'warning'
                st.markdown(f"""
                <div class="kpi-card info">
                    <div class="kpi-label">ADR BED FORECAST 2026</div>
                    <div class="kpi-value">€{metrics.get('ADR_Medio_Forecast', 0):.2f}</div>
                    <div class="kpi-delta">{metrics.get('Variazione_vs_2025', 0):+.1f}% vs 2025</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                gap_class = 'success' if comparison_dashboard.get('gap_room_nights_pct', 0) > 0 else 'critical'
                st.markdown(f"""
                <div class="kpi-card {gap_class}">
                    <div class="kpi-label">ROOM NIGHTS OTB</div>
                    <div class="kpi-value">{comparison_dashboard.get('room_nights_2026', 0):,.0f}</div>
                    <div class="kpi-delta">{comparison_dashboard.get('gap_room_nights_pct', 0):+.1f}% vs LY</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                revenue_total = df_forecast['Revenue_Forecast'].sum()
                st.markdown(f"""
                <div class="kpi-card success">
                    <div class="kpi-label">REVENUE FORECAST</div>
                    <div class="kpi-value">€{revenue_total/1000:.0f}K</div>
                    <div class="kpi-delta">{comparison_dashboard.get('gap_revenue_pct', 0):+.1f}% vs LY</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                occ_media = (df_forecast['Occupazione_Forecast'].mean() * 100) if 'Occupazione_Forecast' in df_forecast.columns else 0
                st.markdown(f"""
                <div class="kpi-card warning">
                    <div class="kpi-label">OCCUPAZIONE MEDIA</div>
                    <div class="kpi-value">{occ_media:.1f}%</div>
                    <div class="kpi-delta">Target: {occupancy_base*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Alert Automatici
            st.subheader("🚨 Alert Automatici")
            
            # Genera alert (per ora senza analisi mensile dettagliata)
            alerts = generate_automatic_alerts(comparison_dashboard, None, df_forecast)
            
            if len(alerts) > 0:
                for alert in alerts:
                    st.markdown(f"""
                    <div class="alert-box {alert['severity']}">
                        <div class="alert-title">{alert['icon']} {alert['title']}</div>
                        <div class="alert-message">{alert['message']}</div>
                        <div class="alert-action">💡 {alert['action']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("✅ Nessun alert critico rilevato")
            
            st.markdown("---")
            
            # NUOVO: Quick Pricing Recommendations
            if pricing_recommendations is not None and len(pricing_recommendations) > 0:
                st.subheader("💰 Top 5 Pricing Actions Needed")
                
                top_5_pricing = pricing_recommendations.head(5)
                
                for idx, rec in top_5_pricing.iterrows():
                    severity_color = {
                        'critical': 'critical',
                        'warning': 'warning',
                        'info': 'info'
                    }.get(rec['Severity'], 'info')
                    
                    st.markdown(f"""
                    <div class="alert-box {severity_color}">
                        <div class="alert-title">{rec['Severity_Icon']} {rec['Data'].strftime('%A %d %B')} - {rec['Giorno_Nome']}</div>
                        <div class="alert-message">
                            <strong>{rec['Action']}</strong><br>
                            ADR Attuale: €{rec['ADR_Current']:.2f} → Target: €{rec['ADR_Recommended']:.2f}<br>
                            Revenue recuperabile: €{rec['Revenue_Gain']:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info("💡 Vai al TAB 'Pricing Recommendations' per l'analisi completa")
            
            st.markdown("---")
            
            # Day-Type Performance
            st.subheader("📊 Performance per Tipo di Giorno")
            
            if day_type_performance is not None and len(day_type_performance) > 0:
                perf = day_type_performance
                
                fig_daytype = go.Figure()
                
                # ADR 2025 vs 2026
                fig_daytype.add_trace(go.Bar(
                    x=perf['Giorno'],
                    y=perf['ADR_2025'],
                    name='2025 Effettivo',
                    marker_color='lightgray'
                ))
                
                fig_daytype.add_trace(go.Bar(
                    x=perf['Giorno'],
                    y=perf['ADR_2026'],
                    name='2026 Forecast',
                    marker_color='#58a6ff'
                ))
                
                fig_daytype.update_layout(
                    title="ADR per Giorno Settimana: 2025 vs 2026",
                    barmode='group',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_daytype, use_container_width=True)
                
                # Tabella performance
                st.dataframe(
                    perf.style.format({
                        'ADR_2025': '€{:.2f}',
                        'ADR_2026': '€{:.2f}',
                        'OCC_2026': '{:.1%}',
                        'Var_%': '{:+.1f}%'
                    }).background_gradient(subset=['Var_%'], cmap='RdYlGn', vmin=-10, vmax=10),
                    use_container_width=True
                )
            
            # NUOVO: Weekly Split Analysis (Dom-Dom per soggiorni 7 notti)
            if weekly_split_analysis is not None and len(weekly_split_analysis) > 0:
                st.markdown("---")
                st.subheader("📅 Analisi Settimanale (Dom-Dom) - Soggiorni 7 Notti")
                
                st.info("""
                **Analisi per settimane Domenica-Domenica** (soggiorno medio 7 notti Bravo)
                
                Confronto:
                - OTB 2026 (snapshot attuale)
                - OTB 2025 stesso periodo (split)
                """)
                
                # Tabella weekly
                display_weekly = weekly_split_analysis.copy()
                
                if 'Data_Inizio' in display_weekly.columns and 'Data_Fine' in display_weekly.columns:
                    display_weekly['Periodo'] = display_weekly.apply(
                        lambda row: f"{row['Data_Inizio'].strftime('%d/%m')} - {row['Data_Fine'].strftime('%d/%m')}",
                        axis=1
                    )
                
                cols_to_show = ['Periodo' if 'Periodo' in display_weekly.columns else 'Settimana']
                
                if 'RN_OTB_2026' in display_weekly.columns:
                    cols_to_show.append('RN_OTB_2026')
                if 'RN_OTB_2025' in display_weekly.columns:
                    cols_to_show.append('RN_OTB_2025')
                if 'Gap_RN' in display_weekly.columns:
                    cols_to_show.append('Gap_RN')
                if 'Gap_RN_Pct' in display_weekly.columns:
                    cols_to_show.append('Gap_RN_Pct')
                if 'ADR_OTB_2026' in display_weekly.columns:
                    cols_to_show.append('ADR_OTB_2026')
                if 'ADR_OTB_2025' in display_weekly.columns:
                    cols_to_show.append('ADR_OTB_2025')
                
                display_df_weekly = display_weekly[cols_to_show].copy()
                
                # Rinomina colonne
                rename_dict = {
                    'Periodo': 'Settimana (Dom-Dom)',
                    'Settimana': 'Settimana',
                    'RN_OTB_2026': 'RN OTB 2026',
                    'RN_OTB_2025': 'RN OTB 2025 (split)',
                    'Gap_RN': 'Gap RN',
                    'Gap_RN_Pct': 'Gap %',
                    'ADR_OTB_2026': 'ADR 2026',
                    'ADR_OTB_2025': 'ADR 2025'
                }
                
                display_df_weekly = display_df_weekly.rename(columns=rename_dict)
                
                # Format
                format_dict = {}
                if 'RN OTB 2026' in display_df_weekly.columns:
                    format_dict['RN OTB 2026'] = '{:,.0f}'
                if 'RN OTB 2025 (split)' in display_df_weekly.columns:
                    format_dict['RN OTB 2025 (split)'] = '{:,.0f}'
                if 'Gap RN' in display_df_weekly.columns:
                    format_dict['Gap RN'] = '{:+,.0f}'
                if 'Gap %' in display_df_weekly.columns:
                    format_dict['Gap %'] = '{:+.1f}%'
                if 'ADR 2026' in display_df_weekly.columns:
                    format_dict['ADR 2026'] = '€{:.2f}'
                if 'ADR 2025' in display_df_weekly.columns:
                    format_dict['ADR 2025'] = '€{:.2f}'
                
                styled_df = display_df_weekly.style.format(format_dict)
                
                if 'Gap %' in display_df_weekly.columns:
                    styled_df = styled_df.background_gradient(
                        subset=['Gap %'],
                        cmap='RdYlGn',
                        vmin=-30,
                        vmax=30
                    )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Insight automatico
                if 'Gap_RN_Pct' in weekly_split_analysis.columns:
                    avg_gap = weekly_split_analysis['Gap_RN_Pct'].mean()
                    
                    if avg_gap < -10:
                        st.error(f"""
                        🔴 **ATTENZIONE:** Gap medio settimanale: {avg_gap:.1f}%
                        
                        Le settimane Domenica-Domenica sono mediamente indietro rispetto allo stesso periodo 2025.
                        Considera promozioni mirate per soggiorni settimanali.
                        """)
                    elif avg_gap < 0:
                        st.warning(f"""
                        🟡 **MONITORAGGIO:** Gap medio settimanale: {avg_gap:.1f}%
                        
                        Lieve ritardo vs 2025. Continua a monitorare il pickup settimanale.
                        """)
                    else:
                        st.success(f"""
                        ✅ **OTTIMO:** Gap medio settimanale: +{avg_gap:.1f}%
                        
                        Le settimane Dom-Dom sono avanti rispetto allo split 2025!
                        """)
        else:
            st.warning("⚠️ Carica gli snapshot 2025 e 2026 per visualizzare il dashboard completo")
    
    # =============================
    # TAB 2: FORECAST 2026
    # =============================
    with tab2:
        st.header("Previsione ADR per Stagione 2026")
        
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
                "ADR Bed Medio 2026",
                f"€{metrics['ADR_Medio_Forecast']:.2f}",
                delta=f"{metrics['Variazione_vs_2025']:.1f}% vs 2025"
            )
            # Aggiungi ADR Cam se disponibile
            if 'ADR_Cam_Forecast' in df_forecast.columns:
                adr_cam_medio = df_forecast['ADR_Cam_Forecast'].mean()
                st.caption(f"ADR Cam: €{adr_cam_medio:.2f}")
        
        with col2:
            st.metric(
                "ADR Bed Medio 2025",
                f"€{metrics['ADR_Medio_2025']:.2f}"
            )
            # Aggiungi ADR Cam 2025 se disponibile nei dati storici
            if 'ADR Cam' in df_historical.columns:
                adr_cam_2025 = df_historical[df_historical['ADR Cam'].notna()]['ADR Cam'].mean()
                st.caption(f"ADR Cam: €{adr_cam_2025:.2f}")
        
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
        
        # Grafico ADR Cam se disponibile
        if 'ADR_Cam_Forecast' in df_forecast.columns and adr_cam_bed_ratio is not None:
            st.markdown("### 🏨 ADR Camera (Room) - Forecast 2026")
            
            fig_adr_cam = go.Figure()
            
            # Storico 2025 ADR Cam se disponibile
            if 'ADR Cam' in df_historical.columns:
                df_2025_cam = df_historical[(df_historical['Anno'] == 2025) & (df_historical['ADR Cam'].notna())].sort_values('Data')
                if len(df_2025_cam) > 0:
                    fig_adr_cam.add_trace(go.Scatter(
                        x=df_2025_cam['Data'],
                        y=df_2025_cam['ADR Cam'],
                        mode='lines+markers',
                        name='2025 Effettivo',
                        line=dict(color='#2ecc71', width=2),
                        marker=dict(size=4)
                    ))
            
            # Forecast 2026 ADR Cam
            fig_adr_cam.add_trace(go.Scatter(
                x=df_forecast['Data'],
                y=df_forecast['ADR_Cam_Forecast'],
                mode='lines+markers',
                name='2026 Forecast',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=5)
            ))
            
            fig_adr_cam.update_layout(
                title=f"ADR Camera Previsto 2026 (Ratio: {adr_cam_bed_ratio:.2f}x vs ADR Bed)",
                xaxis_title="Data",
                yaxis_title="ADR Camera (€)",
                hovermode='x unified',
                height=400,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_adr_cam, use_container_width=True)
        
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
                line=dict(dash='dashdot', color='green')
            ))
            
            # Aggiungi SARIMA se disponibile
            if 'ADR_Bed_SARIMA' in df_forecast.columns:
                fig_models.add_trace(go.Scatter(
                    x=df_forecast['Data'],
                    y=df_forecast['ADR_Bed_SARIMA'],
                    mode='lines',
                    name='SARIMA',
                    line=dict(dash='longdash', color='cyan', width=2)
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
            
            # Info sui pesi del modello
            if 'SARIMA' in models and models['SARIMA'] is not None:
                st.caption("ℹ️ **Ensemble ottimizzato per hotel stagionale:** Random Forest 80% + SARIMA 20% (elimina picchi anomali, cattura stagionalità)")
            else:
                st.caption("ℹ️ **Modello finale:** Random Forest 100% (più stabile per hotel stagionale, elimina picchi anomali del Polynomial)")
        
        # Analisi mensile dettagliata
        st.markdown("### 📅 Analisi Mensile Dettagliata 2026")
        
        # Toggle tra Forecast e Effettivo
        col_toggle, col_spacer = st.columns([1, 3])
        with col_toggle:
            show_actual = st.toggle("📊 Mostra Dati Effettivi (OTB 2026)", value=False, 
                                   help="Passa dalla previsione ai dati reali caricati dal file snapshot 2026")
        
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
        
        # Se l'utente vuole vedere i dati effettivi E sono disponibili
        if show_actual and df_snapshot_2026 is not None:
            # Calcola metriche dai dati OTB reali
            df_snapshot_2026['Mese_Num'] = df_snapshot_2026['Data'].dt.month
            df_snapshot_2026['Mese_Nome'] = df_snapshot_2026['Data'].dt.month_name()
            
            monthly_actual = df_snapshot_2026.groupby(['Mese_Num', 'Mese_Nome']).agg({
                'ADR Bed': ['mean', 'min', 'max', 'std'],
                'Room nights': 'sum',
                'Revenue': 'sum',
                'Data': 'count'
            }).reset_index()
            
            monthly_actual.columns = ['Mese_Num', 'Mese', 'ADR_Medio', 'ADR_Min', 'ADR_Max', 'ADR_StdDev',
                                     'Room_Nights', 'Revenue', 'Giorni']
            
            # Calcola occupazione dai dati reali (se disponibile camere_totali)
            if 'camere_totali' in locals():
                monthly_actual['Occupazione'] = (monthly_actual['Room_Nights'] / 
                                                (camere_totali * monthly_actual['Giorni']) * 100)
            else:
                monthly_actual['Occupazione'] = 0.0
            
            # Usa i dati effettivi
            monthly_data_to_show = monthly_actual.copy()
            st.info("📊 **Visualizzazione Dati Effettivi OTB 2026** (dal file caricato)")
        else:
            # Usa i forecast
            monthly_data_to_show = monthly_forecast.copy()
            if show_actual and df_snapshot_2026 is None:
                st.warning("⚠️ Nessuno snapshot 2026 caricato. Carica un file per vedere i dati effettivi.")
        
        # Calcola metriche mensili per dati storici
        df_historical['Mese_Num'] = df_historical['Data'].dt.month
        
        monthly_2023 = df_historical[df_historical['Anno'] == 2023].groupby('Mese_Num')['ADR Bed'].mean()
        monthly_2024 = df_historical[df_historical['Anno'] == 2024].groupby('Mese_Num')['ADR Bed'].mean()
        monthly_2025 = df_historical[df_historical['Anno'] == 2025].groupby('Mese_Num')['ADR Bed'].mean()
        
        # Aggiungi confronto con anni precedenti
        # Usa i dati selezionati (forecast o effettivo)
        monthly_data_to_show['ADR_2025'] = monthly_data_to_show['Mese_Num'].map(monthly_2025)
        monthly_data_to_show['ADR_2024'] = monthly_data_to_show['Mese_Num'].map(monthly_2024)
        monthly_data_to_show['ADR_2023'] = monthly_data_to_show['Mese_Num'].map(monthly_2023)
        
        # Calcola variazioni
        monthly_data_to_show['Var_vs_2025_%'] = (
            (monthly_data_to_show['ADR_Medio'] - monthly_data_to_show['ADR_2025']) / monthly_data_to_show['ADR_2025'] * 100
        ).round(2)
        
        # Formatta i valori
        monthly_data_to_show['ADR_Medio'] = monthly_data_to_show['ADR_Medio'].round(2)
        monthly_data_to_show['ADR_2025'] = monthly_data_to_show['ADR_2025'].round(2)
        monthly_data_to_show['Room_Nights'] = monthly_data_to_show['Room_Nights'].round(0)
        monthly_data_to_show['Revenue'] = monthly_data_to_show['Revenue'].round(0)
        
        # Formatta occupazione
        if not show_actual:
            monthly_data_to_show['Occupazione'] = (monthly_data_to_show['Occupazione'] * 100).round(1)
        else:
            monthly_data_to_show['Occupazione'] = monthly_data_to_show['Occupazione'].round(1)
        
        # Tabella con tutte le informazioni
        display_df = monthly_data_to_show[['Mese', 'Giorni', 'ADR_Medio', 'Room_Nights', 'Occupazione', 'Revenue',
                                       'ADR_2025', 'Var_vs_2025_%']].copy()
        
        # AGGIUNGI RIGA TOTALE
        totale_row = pd.DataFrame({
            'Mese': ['TOTALE'],
            'Giorni': [display_df['Giorni'].sum()],
            'ADR_Medio': [display_df['ADR_Medio'].mean()],  # Media ADR
            'Room_Nights': [display_df['Room_Nights'].sum()],
            'Occupazione': [display_df['Occupazione'].mean()],  # Media occupazione
            'Revenue': [display_df['Revenue'].sum()],
            'ADR_2025': [display_df['ADR_2025'].mean()],  # Media ADR 2025
            'Var_vs_2025_%': [(display_df['ADR_Medio'].mean() - display_df['ADR_2025'].mean()) / display_df['ADR_2025'].mean() * 100]
        })
        
        display_df = pd.concat([display_df, totale_row], ignore_index=True)
        
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
        
        # Stile per evidenziare riga TOTALE
        def highlight_total(s):
            return ['background-color: #17a2b8; color: white; font-weight: bold' if s['Mese'] == 'TOTALE' else '' for _ in s]
        
        styled_df = display_df.style.format({
            'ADR_Medio': '€{:.2f}',
            'Room_Nights': '{:,.0f}',
            'Occupazione': '{:.1f}%',
            'Revenue': '€{:,.0f}',
            'ADR_2025': '€{:.2f}',
            'Var_vs_2025_%': '{:+.2f}%'
        }).apply(highlight_total, axis=1).map(color_variation, subset=['Var_vs_2025_%'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Grafico comparativo mensile (usa sempre forecast per confronto storico)
        st.markdown("#### 📊 Confronto ADR Mensile: 2023-2026")
        
        fig_monthly_comparison = go.Figure()
        
        # Per il grafico, usa sempre monthly_forecast (che ha dati storici completi)
        # Aggiungi dati storici a monthly_forecast se non ci sono già
        if 'ADR_2023' not in monthly_forecast.columns:
            monthly_forecast['ADR_2025'] = monthly_forecast['Mese_Num'].map(monthly_2025)
            monthly_forecast['ADR_2024'] = monthly_forecast['Mese_Num'].map(monthly_2024)
            monthly_forecast['ADR_2023'] = monthly_forecast['Mese_Num'].map(monthly_2023)
        
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
        
        # Calcola variazioni se non presenti in monthly_plot
        if 'Var_vs_2025_%' not in monthly_plot.columns:
            monthly_plot['Var_vs_2025_%'] = (
                (monthly_plot['ADR_Medio'] - monthly_plot['ADR_2025']) / monthly_plot['ADR_2025'] * 100
            ).round(2)
        
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
    # TAB 3: MODEL ANALYSIS
    # =============================
    with tab3:
        st.header("🔍 Analisi Modelli Predittivi")
        
        # Model Accuracy Metrics
        st.subheader("📈 Metriche di Accuratezza")
        
        if model_metrics and 'RandomForest' in model_metrics:
            rf_metrics = model_metrics['RandomForest']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mape_val = rf_metrics['MAPE']
                mape_status = "🟢 Ottimo" if mape_val < 10 else ("🟡 Buono" if mape_val < 15 else "🔴 Da migliorare")
                st.metric(
                    "MAPE",
                    f"{mape_val:.1f}%",
                    delta=mape_status,
                    help="Mean Absolute Percentage Error - Ottimo se < 10%, Buono se < 15%"
                )
            
            with col2:
                st.metric(
                    "MAE",
                    f"€{rf_metrics['MAE']:.2f}",
                    help="Mean Absolute Error - Errore medio in Euro"
                )
            
            with col3:
                r2_val = rf_metrics['R2']
                r2_status = "🟢 Eccellente" if r2_val > 0.8 else ("🟡 Buono" if r2_val > 0.6 else "🔴 Debole")
                st.metric(
                    "R² Score",
                    f"{r2_val:.3f}",
                    delta=r2_status,
                    help="Varianza spiegata - 1.0 = perfetto, >0.8 = eccellente"
                )
            
            st.markdown("---")
            
            # Feature Importance
            st.subheader("🌲 Random Forest - Feature Importance")
            
            importance_data = rf_metrics['feature_importance']
            
            importance_df = pd.DataFrame({
                'Feature': list(importance_data.keys()),
                'Importance': list(importance_data.values())
            }).sort_values('Importance', ascending=False)
            
            # Traduci feature names
            feature_translations = {
                'Giorno_Stagione': 'Giorno nella Stagione',
                'Settimana_Anno': 'Settimana dell\'Anno',
                'Giorno_Settimana': 'Giorno Settimana (Lun-Dom)',
                'Mese': 'Mese'
            }
            
            importance_df['Feature_IT'] = importance_df['Feature'].map(feature_translations)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature_IT',
                orientation='h',
                title="Quali variabili influenzano di più l'ADR?",
                color='Importance',
                color_continuous_scale='Blues',
                labels={'Feature_IT': 'Variabile', 'Importance': 'Importanza'}
            )
            
            fig_importance.update_layout(
                height=350,
                showlegend=False,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Interpretazione intelligente
            top_feature = importance_df.iloc[0]
            top_feature_name = top_feature['Feature_IT']
            top_feature_pct = top_feature['Importance'] * 100
            
            if top_feature['Feature'] == 'Giorno_Stagione':
                interpretation = f"Il modello si basa principalmente sulla **progressione temporale** della stagione. Questo è normale per un hotel leisure con forte stagionalità."
            elif top_feature['Feature'] == 'Settimana_Anno':
                interpretation = f"Il modello identifica **pattern settimanali ricorrenti** come fattore chiave. Settimane specifiche dell'anno hanno comportamenti ADR distintivi."
            elif top_feature['Feature'] == 'Giorno_Settimana':
                interpretation = f"Il **tipo di giorno** (weekend vs weekday) è il fattore dominante. Questo è tipico per strutture leisure con forte differenziale weekend."
            else:
                interpretation = f"Il **mese** è il fattore principale, indicando macro-stagionalità mensile marcata."
            
            st.info(f"""
            **💡 Interpretazione:**
            
            La variabile più importante è **{top_feature_name}** con un peso del **{top_feature_pct:.1f}%**.
            
            {interpretation}
            """)
            
            st.markdown("---")
            
            # Residual Analysis
            st.subheader("📉 Analisi Residui (Errori del Modello)")
            
            with st.expander("Mostra analisi dettagliata errori", expanded=False):
                # Calcola residui per day of week
                df_hist_copy = df_historical.copy()
                df_hist_copy['Giorno_Nome'] = df_hist_copy['Giorno_Settimana'].map({
                    0: 'Lun', 1: 'Mar', 2: 'Mer', 3: 'Gio', 4: 'Ven', 5: 'Sab', 6: 'Dom'
                })
                
                st.write("**Errore medio per giorno della settimana:**")
                st.write("_(Analisi su dati storici per identificare bias del modello)_")
                
                # Questa analisi richiederebbe di salvare le predizioni durante il training
                # Per ora mostriamo una nota
                st.info("""
                📊 **Analisi Residui Disponibile nei Log di Training**
                
                Per un'analisi completa dei residui:
                - Esporta predizioni vs reali dal training
                - Analizza bias per giorno settimana
                - Identifica periodi con errori sistematici
                """)
        
        else:
            st.warning("⚠️ Metriche modello non disponibili. Verifica che il forecast sia stato generato correttamente.")
        
        st.markdown("---")
        
        # Confronto modelli dettagliato
        st.subheader("📊 Confronto Visivo Tutti i Modelli")
        
        with st.expander("Mostra confronto grafico modelli (Linear, Poly, RF, SARIMA)", expanded=False):
            if show_models and models:
                st.write("**Previsioni di tutti i modelli a confronto:**")
                
                # Questo codice esiste già nel vecchio TAB 1, lo manteniamo qui
                fig_models = go.Figure()
                
                # Dati storici 2025
                df_2025 = df_historical[df_historical['Anno'] == 2025].copy()
                df_2025 = df_2025.sort_values('Data')
                
                fig_models.add_trace(go.Scatter(
                    x=df_2025['Data'],
                    y=df_2025['ADR Bed'],
                    mode='lines',
                    name='2025 Reale',
                    line=dict(color='gray', width=2, dash='dot')
                ))
                
                # Aggiungi ogni modello
                colors = {
                    'Linear': '#3498db',
                    'Polynomial': '#e74c3c', 
                    'RandomForest': '#2ecc71',
                    'SARIMA': '#9b59b6'
                }
                
                for model_name in ['Linear', 'Polynomial', 'RandomForest', 'SARIMA']:
                    if model_name in df_forecast.columns:
                        fig_models.add_trace(go.Scatter(
                            x=df_forecast['Data'],
                            y=df_forecast[model_name],
                            mode='lines',
                            name=model_name,
                            line=dict(color=colors.get(model_name, '#95a5a6'), width=2)
                        ))
                
                fig_models.update_layout(
                    title="Confronto Predizioni: Linear vs Polynomial vs Random Forest vs SARIMA",
                    xaxis_title="Data",
                    yaxis_title="ADR Bed (€)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_models, use_container_width=True)
                
                st.info("""
                **📌 Come leggere il grafico:**
                - **Linear**: Trend lineare semplice
                - **Polynomial**: Cattura curve e non-linearità
                - **Random Forest**: Più flessibile, gestisce interazioni complesse
                - **SARIMA**: Specializzato in serie temporali con stagionalità
                
                Il modello **ensemble finale** usa principalmente Random Forest (80%) + SARIMA (20%) per bilanciare accuratezza e stabilità.
                """)
    
    # =============================
    # TAB 4: BOOKING CURVE & RM (vecchio TAB 5)
    # =============================
    with tab4:
        st.header("Analisi Dati Storici 2023-2025")
        
        # Grafico trend storico ADR Bed
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
            title="ADR Bed: Confronto Stagioni Storiche",
            xaxis_title="Giorno della Stagione",
            yaxis_title="ADR Bed (€)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_historical, use_container_width=True)
        
        # Grafico trend storico ADR Cam se disponibile
        if 'ADR Cam' in df_historical.columns and df_historical['ADR Cam'].notna().any():
            st.markdown("### 🏨 ADR Camera - Trend Storico")
            
            fig_historical_cam = go.Figure()
            
            for anno in [2023, 2024, 2025]:
                df_anno = df_historical[(df_historical['Anno'] == anno) & (df_historical['ADR Cam'].notna())].sort_values('Data')
                if len(df_anno) > 0:
                    fig_historical_cam.add_trace(go.Scatter(
                        x=df_anno['Giorno_Stagione'],
                        y=df_anno['ADR Cam'],
                        mode='lines+markers',
                        name=f'Stagione {anno}',
                        marker=dict(size=4)
                    ))
            
            fig_historical_cam.update_layout(
                title="ADR Camera: Confronto Stagioni Storiche",
                xaxis_title="Giorno della Stagione",
                yaxis_title="ADR Camera (€)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_historical_cam, use_container_width=True)
        
        # Statistiche per anno
        st.markdown("### 📊 Statistiche per Anno")
        
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.markdown("**ADR Bed**")
            stats_by_year = df_historical.groupby('Anno')['ADR Bed'].agg([
                ('Media', 'mean'),
                ('Mediana', 'median'),
                ('Min', 'min'),
                ('Max', 'max'),
                ('Std Dev', 'std')
            ]).round(2)
            st.dataframe(stats_by_year, use_container_width=True)
        
        with col_stats2:
            if 'ADR Cam' in df_historical.columns and df_historical['ADR Cam'].notna().any():
                st.markdown("**ADR Camera**")
                stats_by_year_cam = df_historical[df_historical['ADR Cam'].notna()].groupby('Anno')['ADR Cam'].agg([
                    ('Media', 'mean'),
                    ('Mediana', 'median'),
                    ('Min', 'min'),
                    ('Max', 'max'),
                    ('Std Dev', 'std')
                ]).round(2)
                st.dataframe(stats_by_year_cam, use_container_width=True)
        
        # Distribuzione ADR
        st.markdown("### 📈 Distribuzione ADR Bed")
        
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
        
        # Grafici comparativi per mese in due colonne
        col_bed, col_cam = st.columns(2)
        
        with col_bed:
            st.markdown("**ADR Bed**")
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
                title=f"ADR Bed - {mese_selezionato}",
                xaxis_title="Giorno del Mese",
                yaxis_title="ADR Bed (€)",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_mese, use_container_width=True)
        
        with col_cam:
            if 'ADR Cam' in df_historical.columns and df_mese['ADR Cam'].notna().any():
                st.markdown("**ADR Camera**")
                fig_mese_cam = go.Figure()
                
                for anno in [2023, 2024, 2025]:
                    df_anno_mese_cam = df_mese[(df_mese['Anno'] == anno) & (df_mese['ADR Cam'].notna())].sort_values('Data')
                    if len(df_anno_mese_cam) > 0:
                        fig_mese_cam.add_trace(go.Scatter(
                            x=df_anno_mese_cam['Data'].dt.day,
                            y=df_anno_mese_cam['ADR Cam'],
                            mode='lines+markers',
                            name=f'{anno}',
                            marker=dict(size=6)
                        ))
                
                fig_mese_cam.update_layout(
                    title=f"ADR Cam - {mese_selezionato}",
                    xaxis_title="Giorno del Mese",
                    yaxis_title="ADR Cam (€)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_mese_cam, use_container_width=True)
        
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
        
        adr_bed_mese_2026 = df_mese_2026['ADR_Bed_Forecast'].mean()
        adr_bed_mese_2025 = df_mese_storico[df_mese_storico['Anno'] == 2025]['ADR Bed'].mean() if len(df_mese_storico[df_mese_storico['Anno'] == 2025]) > 0 else 0
        var_mese = ((adr_bed_mese_2026 - adr_bed_mese_2025) / adr_bed_mese_2025 * 100) if adr_bed_mese_2025 > 0 else 0
        
        with col1:
            st.metric(
                "ADR Bed Medio Mese",
                f"€{adr_bed_mese_2026:.2f}",
                delta=f"{var_mese:+.1f}% vs 2025"
            )
            # Aggiungi ADR Cam se disponibile
            if 'ADR_Cam_Forecast' in df_mese_2026.columns:
                adr_cam_mese_2026 = df_mese_2026['ADR_Cam_Forecast'].mean()
                st.caption(f"ADR Cam: €{adr_cam_mese_2026:.2f}")
        
        with col2:
            st.metric(
                "ADR Bed Min",
                f"€{df_mese_2026['ADR_Bed_Forecast'].min():.2f}"
            )
            if 'ADR_Cam_Forecast' in df_mese_2026.columns:
                st.caption(f"Cam: €{df_mese_2026['ADR_Cam_Forecast'].min():.2f}")
        
        with col3:
            st.metric(
                "ADR Bed Max",
                f"€{df_mese_2026['ADR_Bed_Forecast'].max():.2f}"
            )
            if 'ADR_Cam_Forecast' in df_mese_2026.columns:
                st.caption(f"Cam: €{df_mese_2026['ADR_Cam_Forecast'].max():.2f}")
        
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
        revenue_mese = bed_nights_mese * adr_bed_mese_2026
        
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
        
        # Grafici giornalieri del mese - Side by side Bed e Cam
        st.markdown(f"### 📅 Andamento Giornaliero {mese_selezionato}")
        
        col_daily1, col_daily2 = st.columns(2)
        
        with col_daily1:
            st.markdown("**ADR Bed**")
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
            title=f"ADR Bed - {mese_selezionato}",
            xaxis_title="Giorno",
            yaxis_title="ADR Bed (€)",
            hovermode='x unified',
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
        
        with col_daily2:
            if 'ADR_Cam_Forecast' in df_mese_2026.columns and adr_cam_bed_ratio is not None:
                st.markdown("**ADR Camera**")
                fig_daily_cam = go.Figure()
                
                # 2026 Forecast ADR Cam
                fig_daily_cam.add_trace(go.Scatter(
                    x=df_mese_2026_sorted['Data'].dt.day,
                    y=df_mese_2026_sorted['ADR_Cam_Forecast'],
                    mode='lines+markers',
                    name='2026 Forecast',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8),
                    text=df_mese_2026_sorted['Giorno_Nome'],
                    hovertemplate='<b>Giorno %{x}</b><br>%{text}<br>ADR Cam: €%{y:.2f}<extra></extra>'
                ))
                
                # Aggiungi dati storici ADR Cam se disponibili
                if 'ADR Cam' in df_historical.columns:
                    for anno, colore in [(2025, '#2ecc71'), (2024, '#9b59b6'), (2023, '#3498db')]:
                        df_anno_mese_cam = df_mese_storico[(df_mese_storico['Anno'] == anno) & (df_mese_storico['ADR Cam'].notna())].sort_values('Data')
                        if len(df_anno_mese_cam) > 0:
                            fig_daily_cam.add_trace(go.Scatter(
                                x=df_anno_mese_cam['Data'].dt.day,
                                y=df_anno_mese_cam['ADR Cam'],
                                mode='lines+markers',
                                name=f'{anno}',
                                line=dict(color=colore, width=2, dash='dot'),
                                marker=dict(size=6),
                                hovertemplate='<b>Giorno %{x}</b><br>ADR Cam: €%{y:.2f}<extra></extra>'
                            ))
                
                # Evidenzia weekend
                for day in weekend_days:
                    fig_daily_cam.add_vrect(
                        x0=day-0.5, x1=day+0.5,
                        fillcolor="lightblue", opacity=0.1,
                        layer="below", line_width=0
                    )
                
                fig_daily_cam.update_layout(
                    title=f"ADR Cam - {mese_selezionato}",
                    xaxis_title="Giorno",
                    yaxis_title="ADR Cam (€)",
                    hovermode='x unified',
                    height=450,
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_daily_cam, use_container_width=True)
        
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
            # Mostra ADR Bed (sempre disponibile)
            st.metric(
                "ADR Bed 2026",
                f"€{comparison['adr_bed_2026']:.2f}",
                delta=f"€{comparison['gap_adr']:+.2f} ({comparison['gap_adr_pct']:+.1f}%)",
                delta_color="normal"
            )
            
            # Mostra ADR Cam se disponibile
            if comparison['adr_cam_2026'] is not None and comparison['adr_cam_2025'] is not None:
                gap_adr_cam = comparison['adr_cam_2026'] - comparison['adr_cam_2025']
                gap_adr_cam_pct = (gap_adr_cam / comparison['adr_cam_2025'] * 100) if comparison['adr_cam_2025'] > 0 else 0
                st.caption(f"ADR Cam: €{comparison['adr_cam_2026']:.2f} ({gap_adr_cam_pct:+.1f}%)")
        
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
        
        # ADR Breakdown se disponibile
        if comparison['adr_cam_2026'] is not None:
            with st.expander("💰 ADR Breakdown: Bed vs Camera", expanded=False):
                col_adr1, col_adr2, col_adr3 = st.columns(3)
                
                with col_adr1:
                    st.markdown("**2026 OTB**")
                    st.write(f"ADR Bed: €{comparison['adr_bed_2026']:.2f}")
                    st.write(f"ADR Cam: €{comparison['adr_cam_2026']:.2f}")
                    ratio_2026 = comparison['adr_cam_2026'] / comparison['adr_bed_2026']
                    st.write(f"Ratio: {ratio_2026:.2f}x")
                
                with col_adr2:
                    st.markdown("**2025 Comparabile**")
                    st.write(f"ADR Bed: €{comparison['adr_bed_2025']:.2f}")
                    if comparison['adr_cam_2025']:
                        st.write(f"ADR Cam: €{comparison['adr_cam_2025']:.2f}")
                        ratio_2025 = comparison['adr_cam_2025'] / comparison['adr_bed_2025']
                        st.write(f"Ratio: {ratio_2025:.2f}x")
                
                with col_adr3:
                    st.markdown("**Interpretazione**")
                    st.caption(f"Pax medi per camera: ~{ratio_2026:.1f}")
                    if comparison['adr_cam_2025']:
                        gap_cam_pct = ((comparison['adr_cam_2026'] - comparison['adr_cam_2025']) / comparison['adr_cam_2025'] * 100)
                        if gap_cam_pct > 0:
                            st.success(f"ADR Cam +{gap_cam_pct:.1f}%")
                        else:
                            st.error(f"ADR Cam {gap_cam_pct:.1f}%")
        
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
        
        # ROSSO = negativo (sotto target), VERDE = positivo (sopra target)
        colors_gap = ['#e74c3c' if x < 0 else '#27ae60' for x in monthly_comparison['gap_pct']]
        
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
            height=500,  # Aumentato per evitare label tagliate
            template='plotly_white',
            showlegend=False,
            margin=dict(t=80, b=50, l=50, r=50)  # Margini per label
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
    
    # =============================
    # TAB 8: PRICING RECOMMENDATIONS
    # =============================
    with tab8:
        st.header("💰 Raccomandazioni Pricing Intelligenti")
        
        if df_budget_2026 is None:
            st.warning("""
            ⚠️ **Budget 2026 non caricato**
            
            Per generare raccomandazioni pricing automatiche, carica il file budget nella sidebar.
            
            Il sistema analizzerà:
            - Gap giornaliero vs budget
            - ADR target per recuperare revenue
            - Priority days dove intervenire
            - Revenue potenziale recuperabile
            """)
        
        elif pricing_recommendations is None or len(pricing_recommendations) == 0:
            st.success("""
            ✅ **Sei in linea con il budget!**
            
            Nessun adjustment critico necessario. Il forecast attuale è allineato con i target mensili.
            """)
        
        else:
            # Summary KPIs
            st.subheader("📊 Overview Gap vs Budget")
            
            total_gap_revenue = pricing_recommendations['Revenue_Gain'].sum()
            critical_days = len(pricing_recommendations[pricing_recommendations['Severity'] == 'critical'])
            warning_days = len(pricing_recommendations[pricing_recommendations['Severity'] == 'warning'])
            avg_gap_pct = pricing_recommendations['Gap_Pct'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Revenue Gap Recuperabile",
                    f"€{total_gap_revenue:,.0f}",
                    delta=f"{len(pricing_recommendations)} giorni"
                )
            
            with col2:
                st.metric(
                    "Giorni CRITICI",
                    f"{critical_days}",
                    delta="Gap > 20%",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Giorni WARNING",
                    f"{warning_days}",
                    delta="Gap 10-20%",
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "Gap Medio ADR",
                    f"{avg_gap_pct:.1f}%",
                    delta="vs Budget"
                )
            
            st.markdown("---")
            
            # Top Priority Days
            st.subheader("🎯 Top 10 Giorni - Azioni Prioritarie")
            
            top_10 = pricing_recommendations.head(10).copy()
            
            # Tabella raccomandazioni
            top_10['Data_Formatted'] = top_10['Data'].dt.strftime('%a %d/%m')
            
            display_df = top_10[[
                'Severity_Icon', 'Data_Formatted', 'Giorno_Nome',
                'ADR_Current', 'ADR_Storico', 'ADR_Budget', 'ADR_Recommended', 
                'Gap_EUR', 'Gap_Pct', 'Revenue_Gain', 'Action'
            ]].copy()
            
            display_df.columns = [
                '⚠️', 'Data', 'Giorno',
                'ADR Attuale', 'ADR Storico', 'ADR Budget', 'ADR Target',
                'Gap €', 'Gap %', 'Revenue +', 'Azione'
            ]
            
            st.dataframe(
                display_df.style.format({
                    'ADR Attuale': '€{:.2f}',
                    'ADR Storico': '€{:.2f}',
                    'ADR Budget': '€{:.2f}',
                    'ADR Target': '€{:.2f}',
                    'Gap €': '€{:.2f}',
                    'Gap %': '{:.1f}%',
                    'Revenue +': '€{:,.0f}'
                }).background_gradient(subset=['Gap %'], cmap='RdYlGn_r', vmin=-30, vmax=0),
                use_container_width=True,
                height=400
            )
            
            st.markdown("---")
            
            # Detailed Recommendations per Severity
            st.subheader("📋 Raccomandazioni Dettagliate")
            
            tab_critical, tab_warning, tab_info = st.tabs([
                f"🔴 CRITICI ({critical_days})",
                f"🟡 WARNING ({warning_days})",
                f"🔵 INFO ({len(pricing_recommendations) - critical_days - warning_days})"
            ])
            
            with tab_critical:
                critical_recs = pricing_recommendations[pricing_recommendations['Severity'] == 'critical']
                
                if len(critical_recs) > 0:
                    st.write("**Giorni con gap > 20% vs budget - AZIONE IMMEDIATA RICHIESTA**")
                    
                    for idx, rec in critical_recs.head(5).iterrows():
                        with st.expander(f"{rec['Severity_Icon']} {rec['Data'].strftime('%A %d %B')} - Gap {rec['Gap_Pct']:.1f}%", expanded=True):
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                # Mostra tipo raccomandazione
                                rec_type_label = {
                                    'aggressive': '🎯 Target Parziale (Budget Ambizioso)',
                                    'realistic': '✅ Target Realistico',
                                    'conservative': '💚 Target Facile',
                                    'budget_based': '📊 Budget-Based'
                                }.get(rec['Recommendation_Type'], '')
                                
                                st.markdown(f"""
                                **📊 Situazione Attuale:**
                                - ADR Forecast: €{rec['ADR_Current']:.2f}
                                - ADR Budget: €{rec['ADR_Budget']:.2f}
                                - ADR Storico (stesso periodo): €{rec['ADR_Storico']:.2f}
                                - Gap vs Budget: €{abs(rec['Gap_EUR']):.2f} ({abs(rec['Gap_Pct']):.1f}%)
                                - Trend vs Storico: {rec['Trend_vs_Storico']:+.1f}%
                                
                                **🎯 Raccomandazione ({rec_type_label}):**
                                - {rec['Action']}
                                - Revenue recuperabile: €{rec['Revenue_Gain']:,.0f}
                                
                                **💡 Analisi Intelligente:**
                                """)
                                
                                # Analisi intelligente basata su tipo raccomandazione
                                if rec['Recommendation_Type'] == 'aggressive':
                                    st.info("""
                                    ⚠️ Il budget per questo giorno è **molto ambizioso** (+10% vs media storica).
                                    Raccomando target parziale (80% del gap) per evitare resistenza del mercato.
                                    Monitora attentamente il pickup dopo l'aumento.
                                    """)
                                elif rec['Recommendation_Type'] == 'realistic':
                                    st.success("""
                                    ✅ Il budget è **in linea con lo storico**. Target pieno raggiungibile.
                                    Storico dimostra che il mercato accetta questi livelli ADR.
                                    """)
                                elif rec['Recommendation_Type'] == 'conservative':
                                    st.success("""
                                    💚 Il budget è **conservativo** rispetto allo storico. Target facile!
                                    Potresti anche considerare di puntare più in alto.
                                    """)
                                
                                st.markdown("""
                                **🔧 Come Implementare:**
                                1. Chiudi rate code promozionali
                                2. Alza BAR sul channel manager a €{:.2f}
                                3. Limita availability su OTA discount
                                4. Monitora pickup nelle prossime 24-48h
                                5. Se pickup rallenta > 30%, considera rollback parziale
                                """.format(rec['ADR_Recommended']))
                            
                            with col_b:
                                # Mini gauge chart
                                fig_gauge = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=rec['ADR_Current'],
                                    delta={'reference': rec['ADR_Recommended']},
                                    title={'text': "ADR Attuale vs Target"},
                                    gauge={
                                        'axis': {'range': [None, max(rec['ADR_Budget'], rec['ADR_Storico']) * 1.1]},
                                        'bar': {'color': "red"},
                                        'steps': [
                                            {'range': [0, rec['ADR_Storico'] * 0.9], 'color': "lightgray"},
                                            {'range': [rec['ADR_Storico'] * 0.9, rec['ADR_Storico'] * 1.1], 'color': "yellow"},
                                            {'range': [rec['ADR_Storico'] * 1.1, max(rec['ADR_Budget'], rec['ADR_Storico']) * 1.1], 'color': "lightgreen"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "green", 'width': 4},
                                            'thickness': 0.75,
                                            'value': rec['ADR_Recommended']
                                        }
                                    }
                                ))
                                fig_gauge.update_layout(height=250)
                                st.plotly_chart(fig_gauge, use_container_width=True)
                                
                                # Mini chart storico vs budget
                                st.caption(f"📊 Storico: €{rec['ADR_Storico']:.2f}")
                                st.caption(f"🎯 Budget: €{rec['ADR_Budget']:.2f}")
                                st.caption(f"💡 Target: €{rec['ADR_Recommended']:.2f}")
                else:
                    st.success("✅ Nessun giorno critico identificato")
            
            with tab_warning:
                warning_recs = pricing_recommendations[pricing_recommendations['Severity'] == 'warning']
                
                if len(warning_recs) > 0:
                    st.write("**Giorni con gap 10-20% vs budget - Monitoraggio attivo**")
                    
                    # Tabella compatta
                    display_warning = warning_recs[[
                        'Data', 'Giorno_Nome', 'ADR_Current', 'ADR_Recommended', 
                        'Gap_Pct', 'Revenue_Gain', 'Action'
                    ]].copy()
                    
                    display_warning.columns = [
                        'Data', 'Giorno', 'ADR Attuale', 'ADR Target',
                        'Gap %', 'Revenue +', 'Azione'
                    ]
                    
                    st.dataframe(
                        display_warning.style.format({
                            'ADR Attuale': '€{:.2f}',
                            'ADR Target': '€{:.2f}',
                            'Gap %': '{:.1f}%',
                            'Revenue +': '€{:,.0f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.success("✅ Nessun giorno in warning")
            
            with tab_info:
                info_recs = pricing_recommendations[pricing_recommendations['Severity'] == 'info']
                
                if len(info_recs) > 0:
                    st.write("**Giorni con gap < 10% vs budget - Opportunità minori**")
                    
                    # Lista semplice
                    for idx, rec in info_recs.head(10).iterrows():
                        st.write(f"• {rec['Data'].strftime('%d/%m')} ({rec['Giorno_Nome']}): {rec['Action']} → +€{rec['Revenue_Gain']:,.0f}")
                else:
                    st.info("Nessun piccolo adjustment necessario")
            
            st.markdown("---")
            
            # Heatmap Calendar View
            st.subheader("📅 Calendar Heatmap - Gap vs Budget")
            
            # Crea heatmap mensile
            if len(pricing_recommendations) > 0:
                df_heatmap = pricing_recommendations.copy()
                
                # Assicurati che ci siano le colonne necessarie
                if 'Data' in df_heatmap.columns:
                    df_heatmap['Mese'] = df_heatmap['Data'].dt.month
                    df_heatmap['Giorno_Mese'] = df_heatmap['Data'].dt.day
                    df_heatmap['Settimana'] = df_heatmap['Data'].dt.isocalendar().week
                    df_heatmap['Giorno_Settimana_Num'] = df_heatmap['Data'].dt.dayofweek
                    
                    # Verifica che ci siano dati sufficienti
                    if len(df_heatmap) > 0 and 'Gap_Pct' in df_heatmap.columns:
                        # Pivot per heatmap
                        try:
                            # Assicurati che Giorno_Settimana_Num esista
                            if 'Giorno_Settimana_Num' not in df_heatmap.columns:
                                df_heatmap['Giorno_Settimana_Num'] = df_heatmap['Data'].dt.dayofweek
                            
                            pivot_data = df_heatmap.pivot_table(
                                values='Gap_Pct',
                                index='Settimana',
                                columns='Giorno_Settimana_Num',
                                aggfunc='mean'
                            )
                            
                            fig_heatmap = px.imshow(
                                pivot_data,
                                labels=dict(x="Giorno Settimana", y="Settimana", color="Gap %"),
                                x=['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom'],
                                color_continuous_scale='RdYlGn',
                                color_continuous_midpoint=0
                            )
                            
                            fig_heatmap.update_layout(
                                title="Gap % medio per settimana e giorno",
                                height=400
                            )
                            
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        except Exception as e:
                            st.warning(f"Impossibile generare heatmap: {str(e)}")
                    else:
                        st.info("Dati insufficienti per generare heatmap")
                else:
                    st.warning("Colonna 'Data' non trovata nelle raccomandazioni")
    
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
