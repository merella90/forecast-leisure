import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# Configurazione pagina
st.set_page_config(
    page_title="Ca' di Dio Forecast - Revenue Management",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stMetric label {font-size: 14px !important; color: #366092;}
    .stMetric value {font-size: 28px !important; font-weight: bold;}
    h1 {color: #366092;}
    h2 {color: #366092; border-bottom: 2px solid #FFC000; padding-bottom: 10px;}
    h3 {color: #366092;}
    .highlight-box {
        background-color: #FFF9E6;
        border-left: 5px solid #FFC000;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titolo principale
st.title("🏨 Ca' di Dio Vretreats - Forecast Trimestrale Q4/Q1")
st.markdown("**Pickup Method a 4 Componenti** | Revenue Management Dashboard")
st.markdown("---")

# Funzione per caricare dati
@st.cache_data
def load_data(uploaded_files):
    """Carica tutti i dati dai file Excel uploaded"""
    if not uploaded_files or len(uploaded_files) < 5:
        return None
    
    data = {}
    file_mapping = {
        'baseline_2324': None,
        'year_2425': None,
        'otb_2026': None,
        'pickup': None,
        'budget': None
    }
    
    # Identifica i file per nome
    for file in uploaded_files:
        if '150120' in file.name:
            file_mapping['baseline_2324'] = file
        elif '150108' in file.name:
            file_mapping['year_2425'] = file
        elif '145405' in file.name:
            file_mapping['otb_2026'] = file
        elif '145722' in file.name:
            file_mapping['pickup'] = file
        elif '105652' in file.name:
            file_mapping['budget'] = file
    
    try:
        data['baseline_2324'] = pd.read_excel(file_mapping['baseline_2324'])
        data['year_2425'] = pd.read_excel(file_mapping['year_2425'])
        data['otb_2026'] = pd.read_excel(file_mapping['otb_2026'])
        data['pickup'] = pd.read_excel(file_mapping['pickup'])
        data['budget'] = pd.read_excel(file_mapping['budget'])
        
        # Pulizia dati
        for key in ['baseline_2324', 'year_2425', 'otb_2026']:
            df = data[key]
            data[key] = df[df['Giorno'].str.contains('/', na=False) & 
                          ~df['Giorno'].str.contains('Filtri', na=False)].copy()
        
        data['pickup'] = data['pickup'][data['pickup']['Soggiorno'].notna()].copy()
        
        return data
    except Exception as e:
        st.error(f"Errore nel caricamento dati: {e}")
        return None

# Funzione calcolo forecast
def calculate_forecast(baseline, year_prev, otb, pickup, weights, biennale_adj, num_rooms):
    """Calcola il forecast con il metodo a 4 componenti"""
    
    # Estrai pesi
    w_baseline = weights['baseline']
    w_year = weights['year']
    w_otb = weights['otb']
    w_pickup = weights['pickup']
    
    # Calcolo Roomnights Forecast
    rn_forecast = (
        (baseline['rn'] * w_baseline) +
        (year_prev['rn'] * w_year) +
        (otb['rn'] * w_otb) +
        (pickup['rn'] * w_pickup)
    ) * biennale_adj
    
    # Calcolo ADR Forecast (pickup non ha ADR)
    # Riproporziona i pesi dei 3 componenti con ADR
    adr_weight_sum = w_baseline + w_year + w_otb
    adr_forecast = (
        (baseline['adr'] * w_baseline / adr_weight_sum) +
        (year_prev['adr'] * w_year / adr_weight_sum) +
        (otb['adr'] * w_otb / adr_weight_sum)
    ) * biennale_adj
    
    # Calcolo Revenue e Occupancy
    revenue_forecast = rn_forecast * adr_forecast
    occ_forecast = rn_forecast / num_rooms
    
    return {
        'rn': rn_forecast,
        'adr': adr_forecast,
        'revenue': revenue_forecast,
        'occ': occ_forecast
    }

# Sidebar - Upload files
st.sidebar.header("📁 Carica Dati")
uploaded_files = st.sidebar.file_uploader(
    "Carica i 5 file Excel",
    type=['xlsx'],
    accept_multiple_files=True,
    help="Carica: Baseline 2023-24, Year 2024-25, OTB 2026, Pickup, Budget"
)

if not uploaded_files or len(uploaded_files) < 5:
    st.warning("⚠️ Carica tutti i 5 file Excel per procedere")
    st.info("""
    **File necessari:**
    1. Baseline 2023-2024 (Biennale Arte)
    2. Year 2024-2025 (Inflazione)
    3. OTB 2026 (On The Books)
    4. Pickup 7gg (Booking velocity)
    5. Budget con colonne BDG
    """)
    st.stop()

data = load_data(uploaded_files)

if data is None:
    st.error("❌ Errore nel caricamento dei dati")
    st.stop()

st.sidebar.success(f"✅ {len(uploaded_files)} file caricati correttamente")
st.sidebar.markdown("---")

# Sidebar - Parametri
st.sidebar.header("⚙️ Parametri del Modello")

# Parametri hotel
st.sidebar.subheader("Hotel")
num_rooms = st.sidebar.number_input("Camere Disponibili", value=66, min_value=1)

st.sidebar.markdown("---")

# Pesi componenti
st.sidebar.subheader("Pesi Componenti")
st.sidebar.caption("Devono sommare a 100%")

peso_baseline = st.sidebar.slider(
    "① Baseline 2024 (Biennale Arte)",
    min_value=0, max_value=100, value=35, step=5,
    help="Pattern storico stesso ciclo Biennale"
)

peso_year = st.sidebar.slider(
    "② Anno 2025 (Inflazione)",
    min_value=0, max_value=100, value=25, step=5,
    help="Trend ADR recenti e inflazione"
)

peso_otb = st.sidebar.slider(
    "③ OTB 2026",
    min_value=0, max_value=100, value=25, step=5,
    help="Prenotazioni confermate"
)

peso_pickup = st.sidebar.slider(
    "④ Pickup 7gg",
    min_value=0, max_value=100, value=15, step=5,
    help="Booking velocity"
)

# Controllo somma pesi
peso_totale = peso_baseline + peso_year + peso_otb + peso_pickup
if peso_totale != 100:
    st.sidebar.error(f"⚠️ TOTALE: {peso_totale}% (deve essere 100%)")
else:
    st.sidebar.success(f"✅ TOTALE: {peso_totale}%")

weights = {
    'baseline': peso_baseline / 100,
    'year': peso_year / 100,
    'otb': peso_otb / 100,
    'pickup': peso_pickup / 100
}

st.sidebar.markdown("---")

# Biennale Adjustment
st.sidebar.subheader("Biennale Adjustment")
biennale_adj = st.sidebar.number_input(
    "Fattore Moltiplicativo",
    min_value=1.00, max_value=1.50, value=1.10, step=0.01,
    help="Adjustment per Biennale Arte 2026"
)

st.sidebar.markdown("---")

# Bottone reset
if st.sidebar.button("🔄 Reset Parametri Default"):
    st.rerun()

# Tab organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "📈 Grafici", 
    "📋 Dettagli Mesi",
    "🔧 Componenti",
    "💾 Export"
])

# Preparazione dati per calcolo
# DICEMBRE
dic_baseline = {
    'rn': data['baseline_2324'].iloc[0:31]['Room nights'].sum(),
    'adr': data['baseline_2324'].iloc[0:31]['ADR Cam'].mean()
}
dic_year = {
    'rn': data['year_2425'].iloc[0:31]['Room nights'].sum(),
    'adr': data['year_2425'].iloc[0:31]['ADR Cam'].mean()
}
dic_otb = {
    'rn': data['otb_2026'].iloc[0:31]['Room nights'].sum(),
    'adr': data['otb_2026'].iloc[0:31]['ADR Cam'].mean()
}
dic_pickup = {
    'rn': data['pickup'].iloc[0:31]['vs 7gg'].sum() if 'vs 7gg' in data['pickup'].columns else 0,
}

# Actual Dicembre 1-15
dic_actual = {
    'rn': data['otb_2026'].iloc[0:15]['Room nights'].sum(),
    'adr': data['otb_2026'].iloc[0:15]['ADR Cam'].mean(),
    'revenue': data['otb_2026'].iloc[0:15]['Room Revenue'].sum(),
}
dic_actual['occ'] = dic_actual['rn'] / (num_rooms * 15)

# Forecast Dicembre 16-31
dic_16_31_baseline = {
    'rn': data['baseline_2324'].iloc[15:31]['Room nights'].sum(),
    'adr': data['baseline_2324'].iloc[15:31]['ADR Cam'].mean()
}
dic_16_31_year = {
    'rn': data['year_2425'].iloc[15:31]['Room nights'].sum(),
    'adr': data['year_2425'].iloc[15:31]['ADR Cam'].mean()
}
dic_16_31_otb = {
    'rn': data['otb_2026'].iloc[15:31]['Room nights'].sum(),
    'adr': data['otb_2026'].iloc[15:31]['ADR Cam'].mean()
}
dic_16_31_pickup = {
    'rn': data['pickup'].iloc[15:31]['vs 7gg'].sum() if 'vs 7gg' in data['pickup'].columns else 0,
}

dic_16_31_fcst = calculate_forecast(
    dic_16_31_baseline, dic_16_31_year, dic_16_31_otb, 
    dic_16_31_pickup, weights, biennale_adj, num_rooms * 16
)

# Dicembre totale
dic_total = {
    'rn': dic_actual['rn'] + dic_16_31_fcst['rn'],
    'revenue': dic_actual['revenue'] + dic_16_31_fcst['revenue'],
}
dic_total['adr'] = dic_total['revenue'] / dic_total['rn']
dic_total['occ'] = dic_total['rn'] / (num_rooms * 31)

# GENNAIO
gen_baseline = {
    'rn': data['baseline_2324'].iloc[32:63]['Room nights'].sum(),
    'adr': data['baseline_2324'].iloc[32:63]['ADR Cam'].mean()
}
gen_year = {
    'rn': data['year_2425'].iloc[31:62]['Room nights'].sum(),
    'adr': data['year_2425'].iloc[31:62]['ADR Cam'].mean()
}
gen_otb = {
    'rn': data['otb_2026'].iloc[31:62]['Room nights'].sum(),
    'adr': data['otb_2026'].iloc[31:62]['ADR Cam'].mean()
}
gen_pickup = {
    'rn': data['pickup'].iloc[31:62]['vs 7gg'].sum() if 'vs 7gg' in data['pickup'].columns else 0,
}

gen_fcst = calculate_forecast(
    gen_baseline, gen_year, gen_otb, gen_pickup,
    weights, biennale_adj, num_rooms * 31
)

# FEBBRAIO
feb_baseline = {
    'rn': data['baseline_2324'].iloc[63:92]['Room nights'].sum(),
    'adr': data['baseline_2324'].iloc[63:92]['ADR Cam'].mean()
}
feb_year = {
    'rn': data['year_2425'].iloc[62:91]['Room nights'].sum(),
    'adr': data['year_2425'].iloc[62:91]['ADR Cam'].mean()
}
feb_otb = {
    'rn': data['otb_2026'].iloc[62:90]['Room nights'].sum(),
    'adr': data['otb_2026'].iloc[62:90]['ADR Cam'].mean()
}
feb_pickup = {
    'rn': data['pickup'].iloc[62:90]['vs 7gg'].sum() if 'vs 7gg' in data['pickup'].columns else 0,
}

feb_fcst = calculate_forecast(
    feb_baseline, feb_year, feb_otb, feb_pickup,
    weights, biennale_adj, num_rooms * 28
)

# Budget
budget_data = {
    'dic': {
        'rn': data['budget'].iloc[1]['Roomnights BDG'],
        'adr': data['budget'].iloc[1]['ADR Room BDG'],
        'revenue': data['budget'].iloc[1]['Room Revenue BDG'],
        'occ': data['budget'].iloc[1]['Occ.% BDG']
    },
    'gen': {
        'rn': data['budget'].iloc[2]['Roomnights BDG'],
        'adr': data['budget'].iloc[2]['ADR Room BDG'],
        'revenue': data['budget'].iloc[2]['Room Revenue BDG'],
        'occ': data['budget'].iloc[2]['Occ.% BDG']
    },
    'feb': {
        'rn': data['budget'].iloc[3]['Roomnights BDG'],
        'adr': data['budget'].iloc[3]['ADR Room BDG'],
        'revenue': data['budget'].iloc[3]['Room Revenue BDG'],
        'occ': data['budget'].iloc[3]['Occ.% BDG']
    }
}

# TAB 1: DASHBOARD
with tab1:
    st.header("Dashboard KPI")
    
    # Componenti formula
    st.markdown(f"""
    <div class="highlight-box">
    <strong>Formula Pickup Method:</strong> Forecast = [(Baseline 2024 × {peso_baseline}%) + (Anno 2025 × {peso_year}%) + (OTB 2026 × {peso_otb}%) + (Pickup 7gg × {peso_pickup}%)] × {biennale_adj:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # DICEMBRE 2025
    st.subheader("🎄 DICEMBRE 2025")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Roomnights",
            f"{dic_total['rn']:,.0f}",
            f"{((dic_total['rn']/budget_data['dic']['rn']-1)*100):.1f}% vs BDG"
        )
    with col2:
        st.metric(
            "ADR Camera",
            f"€{dic_total['adr']:,.2f}",
            f"{((dic_total['adr']/budget_data['dic']['adr']-1)*100):.1f}% vs BDG"
        )
    with col3:
        st.metric(
            "Room Revenue",
            f"€{dic_total['revenue']:,.0f}",
            f"{((dic_total['revenue']/budget_data['dic']['revenue']-1)*100):.1f}% vs BDG"
        )
    with col4:
        st.metric(
            "Occupancy",
            f"{dic_total['occ']:.1%}",
            f"{(dic_total['occ']-budget_data['dic']['occ']):.1%} vs BDG"
        )
    
    with st.expander("📊 Dettaglio Dicembre: Actual vs Forecast"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Actual 1-15 dic:**")
            st.write(f"RN: {dic_actual['rn']:.0f} | ADR: €{dic_actual['adr']:.2f} | Rev: €{dic_actual['revenue']:,.0f}")
        with col_b:
            st.markdown("**Forecast 16-31 dic:**")
            st.write(f"RN: {dic_16_31_fcst['rn']:.0f} | ADR: €{dic_16_31_fcst['adr']:.2f} | Rev: €{dic_16_31_fcst['revenue']:,.0f}")
    
    st.markdown("---")
    
    # GENNAIO 2026
    st.subheader("❄️ GENNAIO 2026")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Roomnights",
            f"{gen_fcst['rn']:,.0f}",
            f"{((gen_fcst['rn']/budget_data['gen']['rn']-1)*100):.1f}% vs BDG"
        )
    with col2:
        st.metric(
            "ADR Camera",
            f"€{gen_fcst['adr']:,.2f}",
            f"{((gen_fcst['adr']/budget_data['gen']['adr']-1)*100):.1f}% vs BDG"
        )
    with col3:
        st.metric(
            "Room Revenue",
            f"€{gen_fcst['revenue']:,.0f}",
            f"{((gen_fcst['revenue']/budget_data['gen']['revenue']-1)*100):.1f}% vs BDG"
        )
    with col4:
        st.metric(
            "Occupancy",
            f"{gen_fcst['occ']:.1%}",
            f"{(gen_fcst['occ']-budget_data['gen']['occ']):.1%} vs BDG"
        )
    
    st.markdown("---")
    
    # FEBBRAIO 2026
    st.subheader("💝 FEBBRAIO 2026")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Roomnights",
            f"{feb_fcst['rn']:,.0f}",
            f"{((feb_fcst['rn']/budget_data['feb']['rn']-1)*100):.1f}% vs BDG"
        )
    with col2:
        st.metric(
            "ADR Camera",
            f"€{feb_fcst['adr']:,.2f}",
            f"{((feb_fcst['adr']/budget_data['feb']['adr']-1)*100):.1f}% vs BDG"
        )
    with col3:
        st.metric(
            "Room Revenue",
            f"€{feb_fcst['revenue']:,.0f}",
            f"{((feb_fcst['revenue']/budget_data['feb']['revenue']-1)*100):.1f}% vs BDG"
        )
    with col4:
        st.metric(
            "Occupancy",
            f"{feb_fcst['occ']:.1%}",
            f"{(feb_fcst['occ']-budget_data['feb']['occ']):.1%} vs BDG"
        )
    
    st.markdown("---")
    
    # TOTALE TRIMESTRE
    st.subheader("📊 TOTALE TRIMESTRE (Dic 2025 - Feb 2026)")
    
    trim_fcst_rn = dic_total['rn'] + gen_fcst['rn'] + feb_fcst['rn']
    trim_fcst_rev = dic_total['revenue'] + gen_fcst['revenue'] + feb_fcst['revenue']
    trim_fcst_adr = trim_fcst_rev / trim_fcst_rn
    trim_fcst_occ = trim_fcst_rn / (num_rooms * 90)
    
    trim_bdg_rn = budget_data['dic']['rn'] + budget_data['gen']['rn'] + budget_data['feb']['rn']
    trim_bdg_rev = budget_data['dic']['revenue'] + budget_data['gen']['revenue'] + budget_data['feb']['revenue']
    trim_bdg_adr = trim_bdg_rev / trim_bdg_rn
    trim_bdg_occ = trim_bdg_rn / (num_rooms * 90)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Roomnights Totali",
            f"{trim_fcst_rn:,.0f}",
            f"{(trim_fcst_rn - trim_bdg_rn):,.0f} vs BDG",
            delta_color="normal"
        )
    with col2:
        st.metric(
            "ADR Medio",
            f"€{trim_fcst_adr:,.2f}",
            f"€{(trim_fcst_adr - trim_bdg_adr):,.2f} vs BDG",
            delta_color="normal"
        )
    with col3:
        st.metric(
            "Revenue Totale",
            f"€{trim_fcst_rev:,.0f}",
            f"€{(trim_fcst_rev - trim_bdg_rev):,.0f} vs BDG",
            delta_color="normal"
        )
    with col4:
        st.metric(
            "Occupancy Media",
            f"{trim_fcst_occ:.1%}",
            f"{(trim_fcst_occ - trim_bdg_occ):.1%} vs BDG",
            delta_color="normal"
        )

# TAB 2: GRAFICI
with tab2:
    st.header("Grafici Comparativi")
    
    # Prepara dati per grafici
    months = ['Dicembre 2025', 'Gennaio 2026', 'Febbraio 2026']
    
    fcst_rn = [dic_total['rn'], gen_fcst['rn'], feb_fcst['rn']]
    bdg_rn = [budget_data['dic']['rn'], budget_data['gen']['rn'], budget_data['feb']['rn']]
    
    fcst_rev = [dic_total['revenue'], gen_fcst['revenue'], feb_fcst['revenue']]
    bdg_rev = [budget_data['dic']['revenue'], budget_data['gen']['revenue'], budget_data['feb']['revenue']]
    
    fcst_adr = [dic_total['adr'], gen_fcst['adr'], feb_fcst['adr']]
    bdg_adr = [budget_data['dic']['adr'], budget_data['gen']['adr'], budget_data['feb']['adr']]
    
    fcst_occ = [dic_total['occ'], gen_fcst['occ'], feb_fcst['occ']]
    bdg_occ = [budget_data['dic']['occ'], budget_data['gen']['occ'], budget_data['feb']['occ']]
    
    # Grafico Revenue
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        name='Forecast',
        x=months,
        y=fcst_rev,
        marker_color='#366092',
        text=[f'€{v:,.0f}' for v in fcst_rev],
        textposition='auto'
    ))
    fig_rev.add_trace(go.Bar(
        name='Budget',
        x=months,
        y=bdg_rev,
        marker_color='#FFC000',
        text=[f'€{v:,.0f}' for v in bdg_rev],
        textposition='auto'
    ))
    fig_rev.update_layout(
        title='Room Revenue: Forecast vs Budget',
        yaxis_title='Revenue (€)',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_rev, use_container_width=True)
    
    # Grafici KPI side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Grafico Roomnights
        fig_rn = go.Figure()
        fig_rn.add_trace(go.Bar(
            name='Forecast',
            x=months,
            y=fcst_rn,
            marker_color='#366092',
            text=[f'{v:.0f}' for v in fcst_rn],
            textposition='auto'
        ))
        fig_rn.add_trace(go.Bar(
            name='Budget',
            x=months,
            y=bdg_rn,
            marker_color='#FFC000',
            text=[f'{v:.0f}' for v in bdg_rn],
            textposition='auto'
        ))
        fig_rn.update_layout(
            title='Roomnights: Forecast vs Budget',
            yaxis_title='Roomnights',
            barmode='group',
            height=350
        )
        st.plotly_chart(fig_rn, use_container_width=True)
    
    with col2:
        # Grafico ADR
        fig_adr = go.Figure()
        fig_adr.add_trace(go.Bar(
            name='Forecast',
            x=months,
            y=fcst_adr,
            marker_color='#366092',
            text=[f'€{v:.2f}' for v in fcst_adr],
            textposition='auto'
        ))
        fig_adr.add_trace(go.Bar(
            name='Budget',
            x=months,
            y=bdg_adr,
            marker_color='#FFC000',
            text=[f'€{v:.2f}' for v in bdg_adr],
            textposition='auto'
        ))
        fig_adr.update_layout(
            title='ADR Camera: Forecast vs Budget',
            yaxis_title='ADR (€)',
            barmode='group',
            height=350
        )
        st.plotly_chart(fig_adr, use_container_width=True)
    
    # Grafico Occupancy
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(
        name='Forecast',
        x=months,
        y=[v*100 for v in fcst_occ],
        mode='lines+markers',
        line=dict(color='#366092', width=3),
        marker=dict(size=10)
    ))
    fig_occ.add_trace(go.Scatter(
        name='Budget',
        x=months,
        y=[v*100 for v in bdg_occ],
        mode='lines+markers',
        line=dict(color='#FFC000', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    fig_occ.update_layout(
        title='Occupancy %: Forecast vs Budget',
        yaxis_title='Occupancy (%)',
        height=400
    )
    st.plotly_chart(fig_occ, use_container_width=True)

# TAB 3: DETTAGLI MESI
with tab3:
    st.header("Dettagli Mensili")
    
    # Dicembre
    st.subheader("Dicembre 2025")
    df_dic = pd.DataFrame({
        'Metrica': ['Roomnights', 'ADR Camera (€)', 'Room Revenue (€)', 'Occupancy %'],
        'Actual 1-15': [
            f"{dic_actual['rn']:.0f}",
            f"€{dic_actual['adr']:.2f}",
            f"€{dic_actual['revenue']:,.0f}",
            f"{dic_actual['occ']:.1%}"
        ],
        'Forecast 16-31': [
            f"{dic_16_31_fcst['rn']:.0f}",
            f"€{dic_16_31_fcst['adr']:.2f}",
            f"€{dic_16_31_fcst['revenue']:,.0f}",
            f"{dic_16_31_fcst['occ']:.1%}"
        ],
        'Totale': [
            f"{dic_total['rn']:.0f}",
            f"€{dic_total['adr']:.2f}",
            f"€{dic_total['revenue']:,.0f}",
            f"{dic_total['occ']:.1%}"
        ],
        'Budget': [
            f"{budget_data['dic']['rn']:.0f}",
            f"€{budget_data['dic']['adr']:.2f}",
            f"€{budget_data['dic']['revenue']:,.0f}",
            f"{budget_data['dic']['occ']:.1%}"
        ],
        'Var vs BDG': [
            f"{((dic_total['rn']/budget_data['dic']['rn']-1)*100):.1f}%",
            f"{((dic_total['adr']/budget_data['dic']['adr']-1)*100):.1f}%",
            f"{((dic_total['revenue']/budget_data['dic']['revenue']-1)*100):.1f}%",
            f"{(dic_total['occ']-budget_data['dic']['occ']):.1%}"
        ]
    })
    st.dataframe(df_dic, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Gennaio
    st.subheader("Gennaio 2026")
    df_gen = pd.DataFrame({
        'Metrica': ['Roomnights', 'ADR Camera (€)', 'Room Revenue (€)', 'Occupancy %'],
        'Forecast': [
            f"{gen_fcst['rn']:.0f}",
            f"€{gen_fcst['adr']:.2f}",
            f"€{gen_fcst['revenue']:,.0f}",
            f"{gen_fcst['occ']:.1%}"
        ],
        'Budget': [
            f"{budget_data['gen']['rn']:.0f}",
            f"€{budget_data['gen']['adr']:.2f}",
            f"€{budget_data['gen']['revenue']:,.0f}",
            f"{budget_data['gen']['occ']:.1%}"
        ],
        'Var vs BDG': [
            f"{((gen_fcst['rn']/budget_data['gen']['rn']-1)*100):.1f}%",
            f"{((gen_fcst['adr']/budget_data['gen']['adr']-1)*100):.1f}%",
            f"{((gen_fcst['revenue']/budget_data['gen']['revenue']-1)*100):.1f}%",
            f"{(gen_fcst['occ']-budget_data['gen']['occ']):.1%}"
        ]
    })
    st.dataframe(df_gen, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Febbraio
    st.subheader("Febbraio 2026")
    df_feb = pd.DataFrame({
        'Metrica': ['Roomnights', 'ADR Camera (€)', 'Room Revenue (€)', 'Occupancy %'],
        'Forecast': [
            f"{feb_fcst['rn']:.0f}",
            f"€{feb_fcst['adr']:.2f}",
            f"€{feb_fcst['revenue']:,.0f}",
            f"{feb_fcst['occ']:.1%}"
        ],
        'Budget': [
            f"{budget_data['feb']['rn']:.0f}",
            f"€{budget_data['feb']['adr']:.2f}",
            f"€{budget_data['feb']['revenue']:,.0f}",
            f"{budget_data['feb']['occ']:.1%}"
        ],
        'Var vs BDG': [
            f"{((feb_fcst['rn']/budget_data['feb']['rn']-1)*100):.1f}%",
            f"{((feb_fcst['adr']/budget_data['feb']['adr']-1)*100):.1f}%",
            f"{((feb_fcst['revenue']/budget_data['feb']['revenue']-1)*100):.1f}%",
            f"{(feb_fcst['occ']-budget_data['feb']['occ']):.1%}"
        ]
    })
    st.dataframe(df_feb, use_container_width=True, hide_index=True)

# TAB 4: COMPONENTI
with tab4:
    st.header("Breakdown dei 4 Componenti")
    st.write("Visualizza il contributo di ciascun componente al forecast finale.")
    
    # Dicembre 16-31
    st.subheader("Dicembre 2025 (16-31) - Contributi per Roomnights")
    
    contrib_baseline = dic_16_31_baseline['rn'] * weights['baseline'] * biennale_adj
    contrib_year = dic_16_31_year['rn'] * weights['year'] * biennale_adj
    contrib_otb = dic_16_31_otb['rn'] * weights['otb'] * biennale_adj
    contrib_pickup = dic_16_31_pickup['rn'] * weights['pickup'] * biennale_adj
    
    fig_contrib = go.Figure()
    fig_contrib.add_trace(go.Bar(
        x=['Baseline 2024', 'Anno 2025', 'OTB 2026', 'Pickup 7gg'],
        y=[contrib_baseline, contrib_year, contrib_otb, contrib_pickup],
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
        text=[f'{v:.0f}' for v in [contrib_baseline, contrib_year, contrib_otb, contrib_pickup]],
        textposition='auto'
    ))
    fig_contrib.update_layout(
        title=f'Contributo Componenti al Forecast RN (Totale: {dic_16_31_fcst["rn"]:.0f})',
        yaxis_title='Roomnights',
        height=400
    )
    st.plotly_chart(fig_contrib, use_container_width=True)
    
    # Tabella dettagli
    st.markdown("**Dettagli Calcolo Dicembre 16-31:**")
    df_components = pd.DataFrame({
        'Componente': ['① Baseline 2024', '② Anno 2025', '③ OTB 2026', '④ Pickup 7gg', 'TOTALE (prima Biennale Adj)', 'Biennale Adjustment', 'FORECAST FINALE'],
        'Roomnights': [
            dic_16_31_baseline['rn'],
            dic_16_31_year['rn'],
            dic_16_31_otb['rn'],
            dic_16_31_pickup['rn'],
            (dic_16_31_baseline['rn'] * weights['baseline'] + 
             dic_16_31_year['rn'] * weights['year'] +
             dic_16_31_otb['rn'] * weights['otb'] +
             dic_16_31_pickup['rn'] * weights['pickup']),
            f"× {biennale_adj}",
            dic_16_31_fcst['rn']
        ],
        'Peso': [
            f"{weights['baseline']:.1%}",
            f"{weights['year']:.1%}",
            f"{weights['otb']:.1%}",
            f"{weights['pickup']:.1%}",
            "100%",
            "-",
            "-"
        ],
        'Contributo': [
            contrib_baseline,
            contrib_year,
            contrib_otb,
            contrib_pickup,
            "-",
            "-",
            dic_16_31_fcst['rn']
        ]
    })
    st.dataframe(df_components, use_container_width=True, hide_index=True)

# TAB 5: EXPORT
with tab5:
    st.header("Export Risultati")
    
    st.write("Scarica i risultati del forecast in formato Excel per ulteriori analisi.")
    
    # Prepara DataFrame per export
    export_data = pd.DataFrame({
        'Mese': ['Dicembre 2025', 'Gennaio 2026', 'Febbraio 2026', 'TOTALE TRIMESTRE'],
        'Forecast RN': [dic_total['rn'], gen_fcst['rn'], feb_fcst['rn'], trim_fcst_rn],
        'Forecast ADR': [dic_total['adr'], gen_fcst['adr'], feb_fcst['adr'], trim_fcst_adr],
        'Forecast Revenue': [dic_total['revenue'], gen_fcst['revenue'], feb_fcst['revenue'], trim_fcst_rev],
        'Forecast Occ%': [dic_total['occ'], gen_fcst['occ'], feb_fcst['occ'], trim_fcst_occ],
        'Budget RN': [budget_data['dic']['rn'], budget_data['gen']['rn'], budget_data['feb']['rn'], trim_bdg_rn],
        'Budget ADR': [budget_data['dic']['adr'], budget_data['gen']['adr'], budget_data['feb']['adr'], trim_bdg_adr],
        'Budget Revenue': [budget_data['dic']['revenue'], budget_data['gen']['revenue'], budget_data['feb']['revenue'], trim_bdg_rev],
        'Budget Occ%': [budget_data['dic']['occ'], budget_data['gen']['occ'], budget_data['feb']['occ'], trim_bdg_occ],
    })
    
    # Calcola variance
    export_data['Var RN %'] = ((export_data['Forecast RN'] / export_data['Budget RN'] - 1) * 100).round(1)
    export_data['Var ADR %'] = ((export_data['Forecast ADR'] / export_data['Budget ADR'] - 1) * 100).round(1)
    export_data['Var Revenue %'] = ((export_data['Forecast Revenue'] / export_data['Budget Revenue'] - 1) * 100).round(1)
    export_data['Var Occ pp'] = (export_data['Forecast Occ%'] - export_data['Budget Occ%']) * 100
    
    st.dataframe(export_data, use_container_width=True, hide_index=True)
    
    # Parametri usati
    st.markdown("---")
    st.subheader("Parametri Utilizzati")
    params_df = pd.DataFrame({
        'Parametro': [
            'Camere Disponibili',
            'Peso Baseline 2024',
            'Peso Anno 2025',
            'Peso OTB 2026',
            'Peso Pickup 7gg',
            'Biennale Adjustment'
        ],
        'Valore': [
            num_rooms,
            f"{weights['baseline']:.1%}",
            f"{weights['year']:.1%}",
            f"{weights['otb']:.1%}",
            f"{weights['pickup']:.1%}",
            f"{biennale_adj:.2f}"
        ]
    })
    st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    # Export Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_data.to_excel(writer, sheet_name='Forecast Results', index=False)
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
    
    st.download_button(
        label="📥 Scarica Excel",
        data=output.getvalue(),
        file_name=f"Forecast_CaDiDio_4Comp_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Export CSV
    csv = export_data.to_csv(index=False)
    st.download_button(
        label="📥 Scarica CSV",
        data=csv,
        file_name=f"Forecast_CaDiDio_4Comp_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Ca' di Dio Vretreats</strong> - Revenue Management Dashboard</p>
    <p>Pickup Method a 4 Componenti | Powered by Streamlit 🚀</p>
</div>
""", unsafe_allow_html=True)
