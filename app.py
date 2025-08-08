import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

st.title("📈 Time Series Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, dayfirst=True)
    df.columns = [col.strip().lower() for col in df.columns]
    
    df['week starting date'] = pd.to_datetime(df['week starting date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['week starting date'])

    with st.sidebar:
        filter_type = st.radio("Filter by", ("language", "region"))
        if filter_type == "language":
            selected_value = st.selectbox("Select Language", sorted(df['language'].unique()))
            df_filtered = df[df['language'] == selected_value]
        else:
            selected_value = st.selectbox("Select Region", sorted(df['region'].unique()))
            df_filtered = df[df['region'] == selected_value].groupby(['month and year', 'week starting date'], as_index=False)['demand volume'].sum()

        freq = st.radio("View Frequency", ("Weekly", "Monthly"))

        if freq == "Weekly":
            periods = st.slider("Forecast horizon (weeks)", 1, 104, 12)
        else:
            periods = st.slider("Forecast horizon (months)", 1, 30, 12)

        with st.expander("Advanced Prophet Settings"):
            changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.01, 1.0, 0.05, 0.01)
            seasonality_prior_scale = st.slider("Seasonality Prior Scale", 1, 30, 10, 1)
            seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
            yearly_seasonality = st.checkbox("Enable Yearly Seasonality", value=True)

    if st.button("Generate Forecast"):
        prophet_df = df_filtered[['week starting date', 'demand volume']].rename(columns={'week starting date': 'ds', 'demand volume': 'y'})
        if freq == "Monthly":
            prophet_df = prophet_df.resample('MS', on='ds').sum().reset_index()

        m = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality
        )
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=periods, freq='W' if freq == "Weekly" else 'MS')
        forecast = m.predict(future)

        st.subheader("Forecast Plot with Conditional Coloring")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers',
                                 marker=dict(color=['green' if val >= 0 else 'red' for val in forecast['yhat'].diff().fillna(0)]),
                                 name='Forecast'))
        st.plotly_chart(fig, use_container_width=True)

        prophet_df['mom_growth_%'] = prophet_df['y'].pct_change() * 100
        mom_df = prophet_df[['ds', 'y', 'mom_growth_%']].dropna()
        mom_df_styled = mom_df.style.format({'mom_growth_%': "{:.2f}%"}).background_gradient(subset=['mom_growth_%'], cmap='RdYlGn')
        st.write("Month-on-Month % Growth:")
        st.dataframe(mom_df_styled)

        csv_data = forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv_data,
            file_name=f"forecast_{filter_type}_{selected_value}_{freq}.csv",
            mime="text/csv"
        )
