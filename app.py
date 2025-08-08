import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from io import BytesIO

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("📈 Weekly/Monthly Demand Forecast")

# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean date column
    df['Week starting date'] = pd.to_datetime(df['Week starting date'], errors='coerce')
    df = df.dropna(subset=['Week starting date'])

    # Sidebar filters
    st.sidebar.header("Filter & Parameters")

    filter_type = st.sidebar.radio("Filter By", ["Language", "Region"])

    # Decide filter column
    filter_col = "Language" if filter_type == "Language" else "Region"
    filter_options = sorted(df[filter_col].unique())

    # Reset stored value if invalid
    if 'selected_value' not in st.session_state or st.session_state.selected_value not in filter_options:
        st.session_state.selected_value = filter_options[0] if filter_options else None

    selected_filter_val = st.sidebar.selectbox(
        f"Select {filter_type}",
        filter_options,
        index=filter_options.index(st.session_state.selected_value) if st.session_state.selected_value else 0
    )
    st.session_state.selected_value = selected_filter_val

    # Number of weeks forecast
    forecast_weeks = st.sidebar.slider("Forecast Horizon (weeks)", 4, 52, 12)

    # Toggle weekly/monthly
    agg_view = st.sidebar.radio("View", ["Weekly", "Monthly"])

    # Prophet parameter controls
    st.sidebar.subheader("Prophet Parameters")
    changepoints = st.sidebar.slider("n_changepoints", 5, 50, 25)
    yearly_seasonality = st.sidebar.selectbox("Yearly Seasonality", ["auto", True, False])
    weekly_seasonality = st.sidebar.selectbox("Weekly Seasonality", ["auto", True, False])
    seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"])

    # Filtered data
    if filter_type == "Language":
        df_filtered = df[df["Language"] == selected_filter_val]
    else:
        # Aggregate by region
        df_filtered = df[df["Region"] == selected_filter_val]
        df_filtered = df_filtered.groupby(['Month and year', 'Week starting date'], as_index=False)['demand volume'].sum()

    # Aggregate view
    if agg_view == "Monthly":
        df_filtered = df_filtered.groupby(pd.Grouper(key="Week starting date", freq="M")).sum().reset_index()

    # MoM growth grid (Dec → Nov)
    df_filtered['Month'] = df_filtered['Week starting date'].dt.month
    df_filtered['Year'] = df_filtered['Week starting date'].dt.year
    df_mom = df_filtered.groupby(['Year', 'Month'])['demand volume'].sum().reset_index()
    df_mom['MoM Growth %'] = df_mom.groupby('Year')['demand volume'].pct_change() * 100
    month_order = [12] + list(range(1, 12))
    df_mom['Month'] = pd.Categorical(df_mom['Month'], categories=month_order, ordered=True)
    df_mom = df_mom.sort_values(['Year', 'Month'])

    st.subheader("📊 Month-on-Month Growth (Dec → Nov)")
    mom_pivot = df_mom.pivot(index="Year", columns="Month", values="MoM Growth %")
    st.dataframe(mom_pivot.style.format("{:.2f}"))

    # Forecast button
    if st.button("Generate Forecast"):
        prophet_df = df_filtered.rename(columns={"Week starting date": "ds", "demand volume": "y"})
        m = Prophet(
            n_changepoints=changepoints,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            seasonality_mode=seasonality_mode
        )
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=forecast_weeks, freq="W")
        forecast = m.predict(future)

        st.subheader("Forecast Plot")
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)

        # Export CSV
        csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv_data,
            file_name="forecast.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file to start.")
