import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

st.title("📈 Time Series Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, dayfirst=True)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Ensure date column is parsed
    df['week starting date'] = pd.to_datetime(df['week starting date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['week starting date'])

    # Sidebar filters
    filter_type = st.radio("Filter by", ("language", "region"))
    if filter_type == "language":
        selected_value = st.selectbox("Select Language", sorted(df['language'].unique()))
        df_filtered = df[df['language'] == selected_value]
    else:
        selected_value = st.selectbox("Select Region", sorted(df['region'].unique()))
        df_filtered = df[df['region'] == selected_value].groupby(['month and year', 'week starting date'], as_index=False)['demand volume'].sum()
    
    # Toggle for weekly/monthly view
    freq = st.radio("View Frequency", ("Weekly", "Monthly"))

    # Forecast horizon
    periods = st.slider("Forecast horizon (weeks)", 1, 52, 12)

    # Advanced Prophet parameters
    with st.expander("Advanced Prophet Settings"):
        changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.01, 1.0, 0.05, 0.01)
        seasonality_prior_scale = st.slider("Seasonality Prior Scale", 1, 30, 10, 1)
        seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
        yearly_seasonality = st.checkbox("Enable Yearly Seasonality", True)
        weekly_seasonality = st.checkbox("Enable Weekly Seasonality", True)

    if st.button("Generate Forecast"):
        # Prepare data for Prophet
        prophet_df = df_filtered[['week starting date', 'demand volume']].rename(columns={'week starting date': 'ds', 'demand volume': 'y'})
        if freq == "Monthly":
            prophet_df = prophet_df.resample('M', on='ds').sum().reset_index()
        
        # Build model
        m = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality
        )
        m.fit(prophet_df)

        # Forecast
        future = m.make_future_dataframe(periods=periods, freq='W' if freq == "Weekly" else 'M')
        forecast = m.predict(future)

        st.subheader("Forecast Plot")
        st.plotly_chart(m.plot(forecast), use_container_width=True)

        # Yearly % growth table
        prophet_df['year'] = prophet_df['ds'].dt.year
        yearly_growth = prophet_df.groupby('year')['y'].sum().pct_change() * 100
        st.write("Yearly % Growth:")
        st.dataframe(yearly_growth.reset_index().rename(columns={'y': '% growth'}))

        # Download forecast
        csv_data = forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv_data,
            file_name=f"forecast_{filter_type}_{selected_value}_{freq}.csv",
            mime="text/csv"
        )
