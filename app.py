import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="Prophet Forecast App", layout="wide")
st.title("📈 Demand Forecasting App with Prophet")

REQUIRED_COLS = ["Month and year", "Week starting date", "Language", "Region", "Demand volume"]

# ----------------------------
# Session State for persistence
# ----------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "selected_value" not in st.session_state:
    st.session_state.selected_value = None
if "filter_choice" not in st.session_state:
    st.session_state.filter_choice = None

# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error(f"CSV must contain: {REQUIRED_COLS}")
        st.stop()

    # Convert date
    df['Week starting date'] = pd.to_datetime(df['Week starting date'], errors='coerce', dayfirst=True)
    if df['Week starting date'].isnull().any():
        st.error("Invalid dates found in 'Week starting date'.")
        st.stop()

    st.session_state.df = df

# ----------------------------
# If data exists in session
# ----------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    # Filter choice
    st.session_state.filter_choice = st.radio(
        "Filter by:", ["Language", "Region"], horizontal=True,
        index=["Language", "Region"].index(st.session_state.filter_choice) if st.session_state.filter_choice else 0
    )

    if st.session_state.filter_choice == "Language":
        st.session_state.selected_value = st.selectbox(
            "Select Language", sorted(df['Language'].unique()),
            index=0 if not st.session_state.selected_value else sorted(df['Language'].unique()).index(st.session_state.selected_value)
        )
        df_filtered = df[df['Language'] == st.session_state.selected_value]
    else:
        st.session_state.selected_value = st.selectbox(
            "Select Region", sorted(df['Region'].unique()),
            index=0 if not st.session_state.selected_value else sorted(df['Region'].unique()).index(st.session_state.selected_value)
        )
        df_filtered = df[df['Region'] == st.session_state.selected_value]
        df_filtered = df_filtered.groupby('Week starting date', as_index=False)['Demand volume'].sum()

    # View toggle
    view_choice = st.radio("View data as:", ["Weekly", "Monthly"], horizontal=True)
    if view_choice == "Monthly":
        df_filtered['Period'] = df_filtered['Week starting date'].dt.to_period('M').dt.to_timestamp()
        df_filtered = df_filtered.groupby('Period', as_index=False)['Demand volume'].sum()
        df_filtered.rename(columns={'Period': 'ds', 'Demand volume': 'y'}, inplace=True)
        freq = 'M'
    else:
        df_filtered.rename(columns={'Week starting date': 'ds', 'Demand volume': 'y'}, inplace=True)
        freq = 'W'

    # Forecast horizon
    max_period = 52 if freq == 'W' else 12
    horizon = st.slider(
        f"Forecast horizon ({'weeks' if freq == 'W' else 'months'})",
        min_value=4, max_value=max_period, value=12
    )

    # Prophet settings
    yearly_seasonality = st.checkbox("Enable yearly seasonality", value=True)
    weekly_seasonality = st.checkbox("Enable weekly seasonality", value=(freq == 'W'))

    # Fit Prophet
    m = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality)
    m.fit(df_filtered)

    # Forecast
    future = m.make_future_dataframe(periods=horizon, freq=freq)
    forecast = m.predict(future)

    # ----------------------------
    # Plotly chart
    # ----------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['ds'], y=df_filtered['y'],
        mode='lines+markers', name='Actual', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecast', line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,165,0,0)'), name='Confidence Interval'
    ))
    fig.update_layout(
        title=f"{st.session_state.filter_choice}: {st.session_state.selected_value} - {view_choice} Forecast",
        xaxis_title="Date", yaxis_title="Demand Volume"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # MoM Grid (Fiscal: Dec → Nov)
    # ----------------------------
    if view_choice == "Monthly":
        st.subheader("📊 Month-on-Month Volume & Growth by Year (Fiscal: Dec → Nov)")

        df_mom = df_filtered.copy()
        df_mom['Year'] = df_mom['ds'].dt.year
        df_mom['MonthNum'] = df_mom['ds'].dt.month
        df_mom['Month'] = df_mom['ds'].dt.month_name()

        # Month order for fiscal year
        month_order = [12] + list(range(1, 12))
        month_names = [pd.Timestamp(f"2000-{m}-01").strftime('%B') for m in month_order]
        df_mom['Month'] = pd.Categorical(df_mom['Month'], categories=month_names, ordered=True)

        # Fiscal year label: Dec belongs to next fiscal year
        df_mom['FiscalYear'] = df_mom.apply(
            lambda x: x['Year'] + 1 if x['MonthNum'] == 12 else x['Year'], axis=1
        )

        # Pivot for volumes
        pivot_vol = df_mom.pivot_table(
            index='Month', columns='FiscalYear', values='y', aggfunc='sum'
        ).reindex(month_names)

        # Fiscal-aware MoM growth
        pivot_growth = pivot_vol.copy()
        for col in pivot_growth.columns:
            vals = pivot_growth[col].values
            growth = [0]  # Dec has no previous month
            for i in range(1, len(vals)):
                prev = vals[i-1]
                curr = vals[i]
                if pd.isna(prev) or prev == 0:
                    growth.append(0)
                else:
                    growth.append(((curr - prev) / prev) * 100)
            pivot_growth[col] = growth

        st.write("**Monthly Volumes (Fiscal: Dec → Nov)**")
        st.dataframe(pivot_vol.style.format("{:,.0f}"))

        st.write("**Month-on-Month % Growth (Fiscal: Dec → Nov)**")
        st.dataframe(
            pivot_growth.style.format("{:+.1f}%").background_gradient(cmap='RdYlGn', axis=None)
        )

    # ----------------------------
    # Summary metrics
    # ----------------------------
    st.subheader("📌 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Historical Average", round(df_filtered['y'].mean(), 2))
    col2.metric("Forecast Peak", round(forecast['yhat'].max(), 2))
    col3.metric("Forecast Trough", round(forecast['yhat'].min(), 2))

    # ----------------------------
    # Download forecast CSV
    # ----------------------------
    csv_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast Data", csv_forecast, "forecast.csv", "text/csv")

