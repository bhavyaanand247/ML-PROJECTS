import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="DemandFlow AI",
    page_icon="📈",
    layout="wide"
)

model = joblib.load(
    "model/demand_forecasting_model.pkl"
)

df = pd.read_csv(
    "data/train.csv"
)

df["date"] = pd.to_datetime(df["date"])

# FORECASTING FEATURES


df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week_num"] = df["date"].dt.dayofweek
df["week_of_year"] = (
    df["date"].dt.isocalendar().week.astype(int)
)
df["quarter"] = df["date"].dt.quarter

df["lag_1"] = (
    df.groupby(["store", "item"])["sales"]
      .shift(1)
)

df["lag_7"] = (
    df.groupby(["store", "item"])["sales"]
      .shift(7)
)

df["lag_14"] = (
    df.groupby(["store", "item"])["sales"]
      .shift(14)
)

df["lag_28"] = (
    df.groupby(["store", "item"])["sales"]
      .shift(28)
)

df["rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
      .transform(
          lambda x: x.shift(1).rolling(7).mean()
      )
)

df["rolling_mean_28"] = (
    df.groupby(["store", "item"])["sales"]
      .transform(
          lambda x: x.shift(1).rolling(28).mean()
      )
)

# FUTURE FORECASTING FUNCTION


def forecast_future_demand(
    model,
    data,
    store,
    item,
    forecast_days
):

    history = data[
        (data["store"] == store) &
        (data["item"] == item)
    ][["date", "sales"]].copy()

    history = history.sort_values("date")

    predictions = []

    for _ in range(forecast_days):

        next_date = (
            history["date"].max()
            + pd.Timedelta(days=1)
        )

        sales_values = (
            history["sales"]
            .tolist()
        )

        lag_1 = sales_values[-1]
        lag_7 = sales_values[-7]
        lag_14 = sales_values[-14]
        lag_28 = sales_values[-28]

        rolling_mean_7 = np.mean(
            sales_values[-7:]
        )

        rolling_mean_28 = np.mean(
            sales_values[-28:]
        )

        input_data = pd.DataFrame([{
            "store": store,
            "item": item,
            "year": next_date.year,
            "month": next_date.month,
            "day": next_date.day,
            "day_of_week_num": next_date.dayofweek,
            "week_of_year": next_date.isocalendar().week,
            "quarter": next_date.quarter,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_28": lag_28,
            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_28": rolling_mean_28
        }])

        prediction = model.predict(
            input_data
        )[0]

        prediction = max(
            0,
            prediction
        )

        predictions.append({
            "date": next_date,
            "Predicted Demand": round(
                prediction
            )
        })

        new_row = pd.DataFrame({
            "date": [next_date],
            "sales": [prediction]
        })

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )

    return pd.DataFrame(predictions)



st.title("📈 DemandFlow AI")

st.caption(
    "Retail Demand Forecasting & Sales Intelligence"
)

st.divider()

st.success(
    "✓ Forecasting model loaded successfully"
)

# SIDEBAR


with st.sidebar:

    st.markdown("## 🛍️ DemandFlow AI")

    st.caption(
        "Retail Demand Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Demand Prediction",
            "Sales Analytics",
            "Batch Forecasting",
            "Future Forecast",
            "About"
            
        ]
    )

    st.divider()

    st.markdown("### 📦 Dataset")

    st.markdown(
        "**5 Years** of retail sales data"
    )

    st.caption(
        "10 Stores • 50 Products"
    )

# OVERVIEW


if page == "Overview":

    st.markdown("## 📊 Retail Overview")

    st.caption(
        "A high-level view of historical retail demand."
    )

    # KPI calculations

    total_sales = int(df["sales"].sum())

    total_stores = df["store"].nunique()

    total_items = df["item"].nunique()

    avg_daily_sales = (
        df.groupby("date")["sales"]
        .sum()
        .mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Sales",
            f"{total_sales / 1_000_000:.2f}M"
        )

    with col2:
        st.metric(
            "Total Stores",
            total_stores
        )

    with col3:
        st.metric(
            "Total Products",
            total_items
        )

    with col4:
        st.metric(
            "Avg. Daily Sales",
            f"{avg_daily_sales:.0f}"
        )

    
    # SALES BY STORE
    

    st.markdown("### 🏪 Sales Performance by Store")

    store_sales = (
        df.groupby("store")["sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        store_sales,
        x="store",
        y="sales",
        labels={
            "store": "Store",
            "sales": "Total Sales"
        }
    )

    fig.update_traces(
        marker_color="#E76F2E"
    )

    fig.update_layout(
        height=400,
        template="simple_white",
        paper_bgcolor="#F7F4EC",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#18312B"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

  
    # TOP PRODUCTS


    st.markdown("### 📦 Top 10 Best-Selling Products")

    item_sales = (
        df.groupby("item")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        item_sales,
        x="sales",
        y="item",
        orientation="h",
        labels={
            "item": "Product",
            "sales": "Total Sales"
        }
    )

    fig.update_traces(
        marker_color="#5F8D7A"
    )

    fig.update_layout(
        height=450,
        template="simple_white",
        paper_bgcolor="#F7F4EC",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#18312B"
        ),
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# DEMAND PREDICTION


elif page == "Demand Prediction":

    st.markdown("## 🔮 Demand Prediction")

    st.caption(
        "Estimate product demand using historical retail patterns."
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_store = st.selectbox(
            "🏪 Select Store",
            sorted(df["store"].unique())
        )

    with col2:

        selected_item = st.selectbox(
            "📦 Select Product",
            sorted(df["item"].unique())
        )

        st.markdown("### 📅 Select Date")

    selected_date = st.date_input(
        "Choose a date",
        value=df["date"].max().date()
    )

    selected_date = pd.to_datetime(selected_date)

    history = df[
        (df["store"] == selected_store) &
        (df["item"] == selected_item)
    ].sort_values("date")

    selected_row = history[
        history["date"] == selected_date
    ]

    if selected_row.empty:

        st.warning(
            "No historical data is available for this date."
        )

    else:

        row = selected_row.iloc[0]

        input_data = pd.DataFrame([{
            "store": row["store"],
            "item": row["item"],
            "year": row["date"].year,
            "month": row["date"].month,
            "day": row["date"].day,
            "day_of_week_num": row["date"].dayofweek,
            "week_of_year": row["date"].isocalendar().week,
            "quarter": row["date"].quarter,
            "lag_1": row["lag_1"],
            "lag_7": row["lag_7"],
            "lag_14": row["lag_14"],
            "lag_28": row["lag_28"],
            "rolling_mean_7": row["rolling_mean_7"],
            "rolling_mean_28": row["rolling_mean_28"]
        }])

        prediction = model.predict(
            input_data
        )[0]

        st.markdown("### 🎯 Forecast Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Demand",
                f"{prediction:.0f} units"
            )

        with col2:

            st.metric(
                "Actual Sales",
                f"{row['sales']:.0f} units"
            )

        with col3:

            difference = prediction - row["sales"]

            st.metric(
                "Prediction Difference",
                f"{difference:+.0f} units"
            )

# SALES ANALYTICS


elif page == "Sales Analytics":

    st.markdown("## 📊 Sales Analytics")

    st.caption(
        "Explore historical demand patterns across stores, products and time."
    )


    # MONTHLY SALES TREND


    st.markdown("### 📈 Monthly Sales Trend")

    monthly_sales = (
        df.set_index("date")
          .resample("ME")["sales"]
          .sum()
          .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="date",
        y="sales",
        labels={
            "date": "Date",
            "sales": "Total Sales"
        }
    )

    fig.update_traces(
        line_color="#E76F2E"
    )

    fig.update_layout(
        height=450,
        template="simple_white",
        paper_bgcolor="#F7F4EC",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#18312B"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

  
    # STORE AND PRODUCT ANALYSIS

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🏪 Store Performance")

        selected_store = st.selectbox(
            "Select Store",
            sorted(df["store"].unique()),
            key="analytics_store"
        )

        store_data = (
            df[df["store"] == selected_store]
            .groupby("date")["sales"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            store_data,
            x="date",
            y="sales",
            labels={
                "date": "Date",
                "sales": "Sales"
            }
        )

        fig.update_traces(
            line_color="#5F8D7A"
        )

        fig.update_layout(
            height=400,
            template="simple_white",
            paper_bgcolor="#F7F4EC",
            plot_bgcolor="#FFFFFF",
            font=dict(
                color="#18312B"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.markdown("### 📦 Product Performance")

        selected_item = st.selectbox(
            "Select Product",
            sorted(df["item"].unique()),
            key="analytics_item"
        )

        item_data = (
            df[df["item"] == selected_item]
            .groupby("date")["sales"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            item_data,
            x="date",
            y="sales",
            labels={
                "date": "Date",
                "sales": "Sales"
            }
        )

        fig.update_traces(
            line_color="#E76F2E"
        )

        fig.update_layout(
            height=400,
            template="simple_white",
            paper_bgcolor="#F7F4EC",
            plot_bgcolor="#FFFFFF",
            font=dict(
                color="#18312B"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



    st.markdown("### 📅 Average Sales by Day of Week")

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_data = df.copy()

    weekday_data["day_name"] = (
        weekday_data["date"]
        .dt.day_name()
    )

    weekday_sales = (
        weekday_data
        .groupby("day_name")["sales"]
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )

    fig = px.bar(
        weekday_sales,
        x="day_name",
        y="sales",
        labels={
            "day_name": "Day",
            "sales": "Average Sales"
        }
    )

    fig.update_traces(
        marker_color="#5F8D7A"
    )

    fig.update_layout(
        height=400,
        template="simple_white",
        paper_bgcolor="#F7F4EC",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#18312B"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ACTUAL VS PREDICTED DEMAND

    st.markdown("### 🎯 Actual vs Predicted Demand")

    st.caption(
        "Comparison of actual sales and model predictions "
        "over the most recent 30 days."
    )

    forecasting_features = [
        "store",
        "item",
        "year",
        "month",
        "day",
        "day_of_week_num",
        "week_of_year",
        "quarter",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28"
    ]

    comparison_data = df[
        df["date"] >= (
            df["date"].max() - pd.Timedelta(days=30)
        )
    ].copy()

    predictions = model.predict(
        comparison_data[forecasting_features]
    )

    comparison_data["Predicted Demand"] = predictions

    comparison_daily = (
        comparison_data
        .groupby("date")[
            ["sales", "Predicted Demand"]
        ]
        .sum()
        .reset_index()
    )

    fig = px.line(
        comparison_daily,
        x="date",
        y=["sales", "Predicted Demand"],
        markers=True,
        labels={
            "date": "Date",
            "value": "Units",
            "variable": "Type"
        }
    )

    fig.update_traces(
        selector={"name": "sales"},
        line_color="#5F8D7A"
    )

    fig.update_traces(
        selector={"name": "Predicted Demand"},
        line_color="#E76F2E"
    )

    fig.update_layout(
        height=450,
        template="simple_white",
        paper_bgcolor="#F7F4EC",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#18312B"
        ),
        legend_title_text=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# BATCH FORECASTING


elif page == "Batch Forecasting":

    st.markdown("## 📁 Batch Demand Forecasting")

    st.caption(
        "Upload multiple prepared transactions and forecast demand simultaneously."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        st.success(
            f"✓ {len(batch_df):,} rows uploaded successfully."
        )

        st.markdown("### 👀 Data Preview")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        required_features = [
            "store",
            "item",
            "year",
            "month",
            "day",
            "day_of_week_num",
            "week_of_year",
            "quarter",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_28"
        ]

        missing_columns = [
            column
            for column in required_features
            if column not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "❌ Missing required columns:"
            )

            st.write(missing_columns)

        else:

            st.success(
                "✓ All required forecasting features are present."
            )

            forecast_button = st.button(
                "🔮 Forecast Demand",
                use_container_width=True
            )

            if forecast_button:

                prediction_data = batch_df[
                    required_features
                ].copy()

                predictions = model.predict(
                    prediction_data
                )

                results_df = batch_df.copy()

                results_df["Predicted Demand"] = (
                    predictions.round(0).astype(int)
                )

                st.markdown(
                    "### 🎯 Forecast Results"
                )

                st.dataframe(
                    results_df,
                    use_container_width=True
                )


# FUTURE FORECAST

elif page == "Future Forecast":

    st.markdown("## 🔮 Future Demand Forecast")

    st.caption(
        "Forecast upcoming product demand using historical sales patterns."
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_store = st.selectbox(
            "🏪 Select Store",
            sorted(df["store"].unique()),
            key="future_store"
        )

    with col2:

        selected_item = st.selectbox(
            "📦 Select Product",
            sorted(df["item"].unique()),
            key="future_item"
        )

    forecast_days = st.slider(
        "📅 Forecast Horizon",
        min_value=7,
        max_value=30,
        value=14
    )

    st.caption(
        f"Forecasting the next {forecast_days} days"
    )

    generate_forecast = st.button(
        "🚀 Generate Forecast",
        use_container_width=True
    )

    if generate_forecast:

        with st.spinner(
            "DemandFlow AI is generating the forecast..."
        ):

            forecast_df = forecast_future_demand(
                model,
                df,
                selected_store,
                selected_item,
                forecast_days
            )

        st.markdown(
            "### 📈 Forecast Results"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

         
        # FORECAST KPIs

        total_forecast = (
            forecast_df["Predicted Demand"].sum()
        )

        average_forecast = (
            forecast_df["Predicted Demand"].mean()
        )

        peak_demand = (
            forecast_df["Predicted Demand"].max()
        )

        peak_date = forecast_df.loc[
            forecast_df["Predicted Demand"].idxmax(),
            "date"
        ]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📦 Expected Total Demand",
                f"{total_forecast:,.0f} units"
            )

        with col2:

            st.metric(
                "📊 Average Daily Demand",
                f"{average_forecast:,.0f} units"
            )

        with col3:

            st.metric(
                "🔥 Peak Demand",
                f"{peak_demand:,.0f} units"
            )

        st.caption(
            f"Peak demand is expected on {peak_date.strftime('%d %b %Y')}."
        )

        
        # FORECAST CHART

        st.markdown("### 📈 Expected Demand Over the Forecast Period")

        fig = px.line(
            forecast_df,
            x="date",
            y="Predicted Demand",
            markers=True,
            labels={
                "date": "Date",
                "Predicted Demand": "Predicted Units"
            }
        )

        fig.update_traces(
            line_color="#E76F2E",
            marker_color="#5F8D7A"
        )

        fig.update_layout(
            height=450,
            template="simple_white",
            paper_bgcolor="#F7F4EC",
            plot_bgcolor="#FFFFFF",
            font=dict(
                color="#18312B"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



        forecast_csv = forecast_df.to_csv(
            index=False
        ).encode("utf-8")


        # INVENTORY PLANNING

        st.markdown("### 📦 Inventory Planning")

        safety_stock_percentage = st.slider(
            "Safety Stock Buffer",
            min_value=0,
            max_value=50,
            value=15,
            step=5
        )

        recommended_stock = (
            total_forecast
            * (1 + safety_stock_percentage / 100)
        )

        recommended_stock = int(
            np.ceil(recommended_stock)
        )

        inventory_col1, inventory_col2 = st.columns(2)

        with inventory_col1:

            st.metric(
                "Recommended Stock",
                f"{recommended_stock:,} units"
            )

        with inventory_col2:

            buffer_units = (
                recommended_stock
                - int(total_forecast)
            )

            st.metric(
                "Safety Buffer",
                f"{buffer_units:,} units"
            )

        st.info(
            f"Based on the forecast, keeping approximately "
            f"**{recommended_stock:,} units** of inventory "
            f"would provide a **{safety_stock_percentage}% buffer** "
            f"above predicted demand."
        )

        st.download_button(
            label="📥 Download Forecast CSV",
            data=forecast_csv,
            file_name="demand_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )



elif page == "About":

    st.markdown("## ℹ️ About DemandFlow AI")

    st.markdown(
        """
        ### 🛍️ DemandFlow AI

        DemandFlow AI is a retail demand forecasting system
        designed to help businesses understand historical sales
        patterns and estimate upcoming product demand.

        ### 🤖 Machine Learning

        The forecasting engine uses a Random Forest Regressor
        trained on historical retail sales data.

        ### 🧠 Forecasting Features

        The model uses:

        - Store
        - Product
        - Calendar features
        - Previous-day demand
        - Previous-week demand
        - Historical demand lags
        - Rolling demand averages

        ### 📊 What the Dashboard Provides

        - Historical sales analytics
        - Store and product analysis
        - Demand prediction
        - Future demand forecasting
        - Inventory planning estimates
        - CSV export

        ### ⚠️ Important Note

        Forecasts are estimates based on historical patterns.
        Actual future demand can be affected by factors such
        as promotions, holidays, pricing, stock availability,
        and unexpected market conditions.
        """
    )

    st.markdown("### 🤖 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            "6.19 units"
        )

    with col2:
        st.metric(
            "RMSE",
            "8.07 units"
        )

    with col3:
        st.metric(
            "R² Score",
            "0.9203"
        )

    st.caption(
        "Evaluation performed on a time-based test period."
    )

    

    

