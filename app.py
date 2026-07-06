import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils.preprocessing import load_data, clean_data, normalize_data
from utils.indicators import add_indicators
from models.prophet_model import run_prophet
from models.lstm_model import run_lstm
from models.arima_model import run_arima


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="IntelliTrade AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

/* -----------------------------
   APP BACKGROUND
------------------------------*/
.stApp{
    background-color:#0E1117;
}

/* -----------------------------
   MAIN CONTAINER
------------------------------*/
.block-container{
    max-width:100%;
    padding-top:0.8rem;
    padding-bottom:0.8rem;
    padding-left:1.2rem;
    padding-right:1.2rem;
}

/* -----------------------------
   SIDEBAR
------------------------------*/
section[data-testid="stSidebar"]{
    width:280px !important;
    background:#161A28;
}

section[data-testid="stSidebar"] *{
    font-size:16px !important;
}

/* -----------------------------
   MAIN TITLE
------------------------------*/
.main-title{
    text-align:center;
    font-size:46px;
    font-weight:800;
    color:white;
    margin-bottom:0px;
}

/* -----------------------------
   SUBTITLE
------------------------------*/
.subtitle{
    text-align:center;
    font-size:18px;
    color:#BBBBBB;
    margin-bottom:18px;
}

/* -----------------------------
   TABS
------------------------------*/
button[data-baseweb="tab"]{
    font-size:18px !important;
    font-weight:600 !important;
    padding:10px 18px !important;
}

/* -----------------------------
   FILE UPLOADER
------------------------------*/
[data-testid="stFileUploader"]{
    font-size:16px !important;
}

/* -----------------------------
   BUTTONS
------------------------------*/
.stButton button{
    font-size:16px;
    border-radius:8px;
}

/* -----------------------------
   TABLES
------------------------------*/
table{
    width:100%;
    font-size:15px !important;
}

th{
    font-size:16px !important;
}

td{
    font-size:15px !important;
}

/* -----------------------------
   STREAMLIT DATAFRAME
------------------------------*/
[data-testid="stDataFrame"]{
    width:100%;
}

[data-testid="stDataFrame"] div{
    font-size:15px !important;
}

/* -----------------------------
   HEADINGS
------------------------------*/
h1{
    font-size:38px !important;
}

h2{
    font-size:30px !important;
}

h3{
    font-size:24px !important;
}

/* -----------------------------
   METRIC CARDS
------------------------------*/
[data-testid="metric-container"]{
    background:#161A28;
    border-radius:12px;
    padding:12px;
}

/* -----------------------------
   PLOTLY
------------------------------*/
.js-plotly-plot{
    border-radius:12px;
}

/* -----------------------------
   REMOVE EXTRA TOP SPACE
------------------------------*/
div.block-container{
    padding-top:1rem;
}

/* -----------------------------
   HORIZONTAL RULE
------------------------------*/
hr{
    margin-top:10px;
    margin-bottom:10px;
}

/* -----------------------------
   SCROLLBAR
------------------------------*/
::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#444;
    border-radius:10px;
}

::-webkit-scrollbar-track{
    background:#161A28;
}

</style>
""", unsafe_allow_html=True)

# Move this to the top (after set_page_config)
with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.markdown("### Prediction Days")

    days = st.slider(
        "",
        10,
        60,
        30,
        step=5
    )

    st.markdown("### Select CSV File")

    show_raw = st.checkbox("Show Raw Data", True)

st.markdown("""
<h1 class="main-title">
📈 IntelliTrade AI: Market Forecasting
</h1>

<p class="subtitle">
AI-Powered Stock Market Prediction using Prophet, LSTM & ARIMA
</p>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER (BIGGER + CLEAN)
# -------------------------------


st.markdown("---")

# -------------------------------
# FILE UPLOADER (CENTERED)
# -------------------------------
# Wrap your uploader in a container for better visual hierarchy
upload_container = st.container()
with upload_container:
    file = st.file_uploader("📂 Upload CSV File", type=["csv"])

st.markdown("---")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if file is not None:

    # =========================
    # DATA PROCESSING
    # =========================
    df = load_data(file)
    df = clean_data(df)
    df = add_indicators(df)
    tab1, tab2, tab3 = st.tabs([
    "📊 Raw Data",
    "📈 Predictions",
    "📉 Analysis"
])

    with tab1:

        st.markdown("### 📊 Raw Data")

        st.dataframe(
            df.tail(20),
            use_container_width=True,
            height=450
        )

    st.markdown("---")

    # =========================
    # PROPHET
    # =========================
    with tab2:
        st.markdown("### 🔮 Prophet Prediction")
        st.warning("Prophet temporarily disabled.")

#         forecast = run_prophet(df).tail(60)

#         fig1 = go.Figure()
#         fig1.add_trace(go.Scatter(
#         x=forecast['ds'],
#         y=forecast['yhat'],
#         mode='lines',
#         line=dict(width=5),
#         name='Forecast'
# ))
        
#         fig1.update_layout(
#             template="plotly_dark",

#             height=420,

#             title=dict(
#                 text="Prophet Forecast",
#                 font=dict(size=22)
#             ),

#             xaxis=dict(
#                 title_font=dict(size=16),
#                 tickfont=dict(size=12)
#             ),

#             yaxis=dict(
#                 title_font=dict(size=16),
#                 tickfont=dict(size=12)
#             ),

#             legend=dict(
#                 font=dict(size=13)
#             ),

#             font=dict(size=14),

#             margin=dict(
#                 l=10,
#                 r=10,
#                 t=45,
#                 b=10
#             )
#         )

#         st.plotly_chart(fig1, use_container_width=True)

#         st.markdown("---")
        # =========================
        # LSTM PREDICTION
        # =========================

        st.markdown("### 🧠 LSTM Prediction")

        scaled_data, scaler = normalize_data(df)

        future_preds = run_lstm(scaled_data, scaler)

        future_preds = np.array(future_preds).flatten()

        x_lstm = np.arange(len(future_preds))

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=x_lstm,
            y=future_preds,
            mode='lines+markers',
            name='LSTM Forecast',
            line=dict(color='cyan', width=5),
            marker=dict(size=5)
        ))

        fig2.update_layout(
            template="plotly_dark",

            height=350,

            title=dict(
                text="LSTM Forecast",
                font=dict(size=22)
            ),

            xaxis=dict(
                title="Days",
                title_font=dict(size=16),
                tickfont=dict(size=12)
            ),

            yaxis=dict(
                title="Price",
                title_font=dict(size=16),
                tickfont=dict(size=12)
            ),

            legend=dict(
                font=dict(size=13)
            ),

            font=dict(size=14),

            margin=dict(
                l=10,
                r=10,
                t=45,
                b=10
            )
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")


        # =========================
        # ARIMA PREDICTION
        # =========================

        st.markdown("### 📉 ARIMA Prediction")

        arima_preds = run_arima(df)

        arima_preds = np.array(arima_preds).flatten()

        x_arima = np.arange(len(arima_preds))

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=x_arima,
            y=arima_preds,
            mode='lines+markers',
            name='ARIMA Forecast',
            line=dict(color='orange', width=5),
            marker=dict(size=5)
        ))

        fig3.update_layout(
            template="plotly_dark",

            height=380,

            title=dict(
                text="ARIMA Forecast",
                font=dict(size=22)
            ),

            xaxis=dict(
                title="Days",
                title_font=dict(size=16),
                tickfont=dict(size=12)
            ),

            yaxis=dict(
                title="Price",
                title_font=dict(size=16),
                tickfont=dict(size=12)
            ),

            legend=dict(
                font=dict(size=13)
            ),

            font=dict(size=14),

            margin=dict(
                l=10,
                r=10,
                t=45,
                b=10
            )
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")


        # =========================
        # MODEL COMPARISON
        # =========================

        st.markdown("### 📊 Model Comparison")

        prophet_vals = np.array(
            forecast['yhat'][-30:]
            
        ).flatten()

        min_length = min(
            len(prophet_vals),
            len(future_preds),
            len(arima_preds)
        )

        prophet_vals = prophet_vals[:min_length]
        future_preds = future_preds[:min_length]
        arima_preds = arima_preds[:min_length]

        x_compare = np.arange(min_length)

        fig4 = go.Figure()

        # Prophet
        fig4.add_trace(go.Scatter(
            x=x_compare,
            y=prophet_vals,
            mode='lines+markers',
            name='Prophet',
            line=dict(color='lime', width=5),
            marker=dict(size=5)
        ))

        # LSTM
        fig4.add_trace(go.Scatter(
            x=x_compare,
            y=future_preds,
            mode='lines+markers',
            name='LSTM',
            line=dict(color='cyan', width=5),
            marker=dict(size=5)
        ))

        # ARIMA
        fig4.add_trace(go.Scatter(
            x=x_compare,
            y=arima_preds,
            mode='lines+markers',
            name='ARIMA',
            line=dict(color='orange', width=5),
            marker=dict(size=5)
        ))

        fig4.update_layout(
            template="plotly_dark",

            height=430,

            title=dict(
                text="Model Comparison Forecast",
                font=dict(size=22)
            ),

            xaxis=dict(
                title="Days",
                title_font=dict(size=16),
                tickfont=dict(size=12)
            ),

            yaxis=dict(
                title="Predicted Price",
                title_font=dict(size=16),
                tickfont=dict(size=12)
            ),

            legend=dict(
                font=dict(size=13)
            ),

            font=dict(size=14),

            margin=dict(
                l=10,
                r=10,
                t=45,
                b=10
            )
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

    # =========================
    # TRADING SIGNALS
    # =========================
    with tab3:

        st.markdown("### 💰 Trading Signals")

        df['Signal'] = 'Hold'
        df.loc[df['RSI'] < 30, 'Signal'] = 'Buy'
        df.loc[df['RSI'] > 70, 'Signal'] = 'Sell'

        st.dataframe(
            df[['Close', 'RSI', 'Signal']].tail(20),
            use_container_width=True
        )

else:
    st.info("📂 Please upload a CSV file to start.")