import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from io import BytesIO
import calendar

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

st.title("📈 Time Series Forecasting App (Prophet)")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Date parsing
    try:
        df['Week starting date'] = pd.to_datetime(df['Week starting date'], errors='coerce', dayfirst=True)
    except:
        st.error("Invalid dates found in 'Week starting date'. Please check the file.")
        st.stop()

    if df['Week starting date'].isna().any():
        st.error("Some dates in 'Week starting date' are invalid or missing.")
        st.stop()

    # Ensure column names match expected
    expected_cols = {'Month and year', 'Week starting date', 'language', 'region', 'demand volume'}
    if not expected_cols.issubset(set(df.columns.str.lower())):
        st.warning(f"CSV must contain columns: {expected_cols}")
        st.stop()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Sidebar filters
    st.sidebar.header("Filters")
    filter_type = st.sidebar.radio("Filter by:", ["Language", "Region"], index=0)
    selected_value = None

    if filter_type == "Language":
        lang_options = sorted(df['language'].dropna().unique())
        selected_value = st.sidebar.selectbox("Select Language", ["All"] + lang_options)
        if selected_value != "All":
            df_filtered = df[df['language'] == selected_value]
        else:
            df_filtered = df.copy()
    else:
        region_options = sorted(df['region'].dropna().unique())
        selected_value = st.sidebar.selectbox("Select Region", ["All"] + region_options)
        if selected_value != "All":
            df_filtered = df[df['region'] == selected_value]
        else:
            df_filtered = df.copy()
        # Aggregate if filtering by region
        df_filtered = df_filtered.groupby(['month and year', 'week starting date'], as_index=False)['demand volume'].sum()

    if df_filtered.empty:
        st.warning("No data available for the selected filter.")
        st.stop()

    # Weekly / Monthly toggle
    view_type = st.sidebar.radio("View data as:", ["Weekly", "Monthly"], index=0)

    if view_type == "Monthly":
        df_filtered['month'] = df_filtered['week starting date'].dt.month
        df_filtered['year'] = df_filtered['week starting date'].dt.year
        df_filtered = df_filtered.groupby(['year', 'month'], as_index=False)['demand volume'].sum()
        df_filtered['date'] = pd.to_datetime(df_filtered[['year', 'month']].assign(day=1))
    else:
        df_filtered['date'] = df_filtered['week starting date']

    # Prophet parameters - advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings (Prophet Parameters)"):
        changepoint_prior_scale = st.slider(
            "Changepoint Prior Scale", 0.01, 1.0, 0.05, 0.01
        )
        seasonality_prior_scale = st.slider(
            "Seasonality Prior Scale", 1, 30, 10, 1
        )
        seasonality_mode = st.selectbox(
            "Seasonality Mode", ["additive", "multiplicative"], index=0
        )

    # Forecast horizon
    forecast_weeks = st.sidebar.number_input(
        "Forecast Horizon (weeks)", min_value=1, max_value=104, value=12
    )

    # Generate forecast button
    if st.button("Generate Forecast"):
        prophet_df = df_filtered[['date', 'demand volume']].rename(columns={'date': 'ds', 'demand volume': 'y'})

        # Fit model
        m = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode
        )
        m.fit(prophet_df)

        # Make future dataframe
        future = m.make_future_dataframe(periods=forecast_weeks, freq='W' if view_type == "Weekly" else 'MS')
        forecast = m.predict(future)

        # Plot
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)

        # Download CSV
        csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        filename = f"forecast_{filter_type.lower()}_{selected_value if selected_value != 'All' else 'all'}_{view_type.lower()}.csv"
        st.download_button("Download Forecast CSV", csv_data, file_name=filename, mime='text/csv')

        # Yearly Month-on-Month % growth
        df_growth = df_filtered.copy()
        df_growth['month'] = df_growth['date'].dt.month
        df_growth['year'] = df_growth['date'].dt.year
        monthly_agg = df_growth.groupby(['year', 'month'], as_index=False)['demand volume'].sum()

        # Shift by month within each year for growth
        monthly_agg['pct_growth'] = monthly_agg.groupby('year')['demand volume'].pct_change() * 100
        monthly_agg = monthly_agg.dropna()

        # Sort starting from December → November
        month_order = [12] + list(range(1, 12))
        monthly_agg['month'] = pd.Categorical(monthly_agg['month'], categories=month_order, ordered=True)
        monthly_agg = monthly_agg.sort_values(['year', 'month'])

        # Pivot for display
        growth_pivot = monthly_agg.pivot(index='month', columns='year', values='pct_growth')
        growth_pivot.index = [calendar.month_abbr[m] for m in growth_pivot.index]

        st.subheader("📊 Yearly Month-on-Month % Growth (Dec → Nov)")
        st.dataframe(growth_pivot.style.format("{:.2f}%"))

